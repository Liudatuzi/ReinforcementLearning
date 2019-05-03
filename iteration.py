# 声明状态
states = [i for i in range(16)]
# 声明状态价值，并初始化各状态价值为0
values = [0 for _ in range(16)]
actions = ["N", "E", "S", "W"]
# 声明行为对状态的改变,move north then the current state -4
ds_actions = {"N": -4, "E": 1, "S": 4, "W": -1}
gramma = 1.00  # 衰减系数
a = []
for i in range(16):
    a.append(actions)
#determin the next state according to the current state with the action
def nextState(s,a):
    next_state=s
    #边界情况
    if(s%4==0 and a=="W")or (s<4 and a=="N") or ((s+1)%4==0 and a=="E")or \
    (s>11 and a=="S"):
        pass
    else:
        ds=ds_actions[a]
        next_state=s+ds
    return next_state

#get the reward when leaving the state
def rewardOf(s):
    return 0 if s in [0,15] else -1



#check whether the terminatestate
def isTerminateState(s):
    return s in [0,15]


def printValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]),end = " ")
        if(i+1)%4==0:
            print("")
    print()


#策略
def allowedActions(s):
    allow=[]
    candidate=[]
    if isTerminateState(s):
        return allow
    for a in actions:
        next_state=nextState(s,a)
        candidate.append(values[next_state])
    b=max(candidate)
    for a in actions:
        next_state=nextState(s,a)
        if(values[next_state]==b):
            allow.append(a)
    return allow

#获取某一个状态所有可能的后继状态,根据更新的策略
def allowedStates(s):
    allowStates=[]
    if isTerminateState(s):
        return allowStates
    for a in allowedActions(s):
        next_state=nextState(s,a)
        allowStates.append(next_state)
    return allowStates

def allowedUpdateValue(s):
    next_state=allowedStates(s)
    newValue=0
    num=len(next_state)
    reward=rewardOf(s)
    for state in next_state:
        newValue+=1.00/num*(reward+gramma*values[state])
    return newValue

#迭代并更新策略
def allowedPerformOneIteration():
    newValues=[0 for _ in range(16)]
    newa=[]
    for s  in states:
        allowedActions(s)
        newa.append(allowedActions(s))
    global a
    a=newa
    #策略更新完毕
    for s  in states:
        newValues[s]=allowedUpdateValue(s)
    global values
    values=newValues
    printValue(values)

def main():
    max_iterate_times=10
    cur_iterate_times=0
    while cur_iterate_times<=max_iterate_times:
        print("Iterate No.{0}".format(cur_iterate_times))
        allowedPerformOneIteration()
        #performOneIteration()
        cur_iterate_times+=1
    printValue(values)
    for i in range(16):
        print(allowedActions(i))


if __name__=='__main__':
    main()