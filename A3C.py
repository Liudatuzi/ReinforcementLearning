import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import math, os
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"
update_global_iter=5
gamma=0.9
MAX_EP=3000
MAX_EP_STEP=200

env=gym.make('Pendulum-v0')
N_S=env.observation_space.shape[0]
N_A=env.action_space.shape[0]

def translate_dtype(np_array,dtype=np.float32):
    if np_array.dtype!=dtype:
        np_array=np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight,mean=0.,std=0.1)
        nn.init.constant_(layer.bias,0.)

def push_pull(opt,local_net,global_net,done,s,bs,ba,br,gamma):
    #push means the local_net send their parameters to the global net
    #pull means the local_net grab parameters from the global net
    if done:
        v_s=0.
    else:
        v_s=local_net.forward(translate_dtype(s[None,:]))[-1].data.numpy()[0,0]
    #reverse buffer r
    buffer_v_target=[]
    for r in br[::-1]:
        v_s=r+gamma*v_s
        buffer_v_target.append(v_s)
    buffer_v_target.reverse()

    loss=local_net.loss_func(
        translate_dtype(np.vstack(bs)),
        translate_dtype(np.array(ba),dtype=np.int64) if ba[0].dtype==np.int64  else translate_dtype(np.vstack(ba)),
        translate_dtype(np.array(buffer_v_target)[:,None]))
    #calculate the local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp,gp in zip(local_net.parameters(),global_net.parameters()):
        gp._grad=lp.grad
    opt.step()
    #pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep,global_ep_r,ep_r,res_queue,name):
    with global_ep.get_lock():
        global_ep.value+=1
    with global_ep_r.get_lock():
        if global_ep_r.value==0.:
            global_ep_r.value=ep_r
        else:
            global_ep_r.value=global_ep_r.value*0.99+ep_r*0.01
    res_queue.put(global_ep_r.value)
    print(name,"Ep:",global_ep.value,"Ep_r",global_ep_r.value)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
class Net(nn.Module):
    def __init__(self,space_dim,action_dim):
        super(Net, self).__init__()
        self.space_dim=space_dim
        self.action_dim=action_dim
        self.a1=nn.Linear(space_dim,200)
        self.mu=nn.Linear(200,action_dim)
        self.sigma=nn.Linear(200,action_dim)

        self.c1=nn.Linear(space_dim,100)
        self.v=nn.Linear(100,1)

        set_init([self.a1,self.mu,self.sigma,self.c1,self.v])
        self.distribution=torch.distributions.Normal
    def forward(self, x):
        a1=F.relu6(self.a1(x))
        mu=2*F.tanh(self.mu(a1))
        sigma=F.softplus(self.sigma(a1))+0.001
        c1=F.relu6(self.c1(x))
        values=self.v(c1)
        return mu,sigma,values
    def choose_action(self,s):
        self.training=False
        mu,sigma,values=self.forward(s)
        m=self.distribution(mu.view(1,).data,sigma.view(1,).data)
        return m.sample().numpy()
    def loss_func(self,s,a,v_t):
        self.train()
        mu,sigma,values=self.forward(s)
        td=v_t-values
        c_loss=td.pow(2)
        m=self.distribution(mu,sigma)
        log_prob=m.log_prob(a)
        entropy=0.5+0.5*math.log(2*math.pi)+torch.log(m.scale)
        exp_v=log_prob*td.detach()+0.005*entropy
        a_loss=-exp_v
        totol_loss=(a_loss+c_loss).mean()
        return totol_loss


class Worker(mp.Process):
    def __init__(self,global_net,opt,global_ep,global_ep_r,res_queue,name):
        super(Worker,self).__init__()
        self.name='local_agent%i'%name
        self.g_ep,self.g_ep_r,self.res_queue=global_ep,global_ep_r,res_queue
        self.gnet,self.opt=global_net,opt
        self.lnet=Net(N_S,N_A)
        self.env=gym.make('Pendulum-v0').unwrapped
    def run(self):
        total_step=1
        while self.g_ep.value<MAX_EP:
            s=self.env.reset()
            buffer_s,buffer_a,buffer_r=[],[],[]
            ep_r=0.
            for t in range(MAX_EP_STEP):
                if self.name=='local_agent0':
                    self.env.render()
                a=self.lnet.choose_action(translate_dtype(s[None,:]))
                s_,r,done,_=self.env.step(a.clip(-2,2))
                if t==MAX_EP_STEP-1:
                    done=True
                ep_r+=r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)
                #update to global and assign to local net
                if total_step%update_global_iter==0 or done:
                    push_pull(opt=self.opt,local_net=self.lnet,global_net=self.gnet,done=done,s=s_,bs=buffer_s,ba=buffer_a,br=buffer_r,gamma=gamma)
                    buffer_r,buffer_a,buffer_s=[],[],[]
                    if done:
                        record(self.g_ep,self.g_ep_r,ep_r,self.res_queue,self.name)
                        break
                s=s_
                total_step+=1
        self.res_queue.put(None)

if __name__=="__main__":
    # global network
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=0.0002)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.figure()
    x = [i for i in range(len(res))]
    plt.plot(x, res, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Moving average episode reward')
    plt.show()




