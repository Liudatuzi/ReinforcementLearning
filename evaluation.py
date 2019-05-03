#声明状态
states=[i for i in range(16)]
#声明状态价值，并初始化各状态价值为0
values=[0 for _ in range(16) ]
actions=["N","E","S","W"]
#声明行为对状态的改变,move north then the current state -4
ds_actions={"N":-4,"E":1,"S":4,"W":-1}
gramma=1.00#衰减系数
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

#get all the next states
def getAllState(s):
    next=[]
    if isTerminateState(s):
        return next
    for a in actions:
        next_state=nextState(s,a)
        next.append(next_state)#include not move
    return next

#update the current value
def updateValue(s):
    AllState=getAllState(s)
    newValue=0
    num=4
    reward=rewardOf(s)#leaving reward
    for next_state in AllState:
        newValue+=1.00/num*(reward+gramma*values[next_state])
    return newValue


def printValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]), end=" ")
        if (i + 1) % 4 == 0:
            print("")
    print()


def performOneIteration():
    newValues=[0 for _ in range(16)]
    for s in states:
        newValues[s]=updateValue(s)
    global values
    values=newValues
    printValue(values)

def main():
    max_iterate_times=160
    cur_iterate_times=0
    while cur_iterate_times<=max_iterate_times:
        print("Iterate No.{0}".format(cur_iterate_times))
        performOneIteration()
        cur_iterate_times+=1
    printValue(values)

if __name__=='__main__':
    main()