import random

# 声明状态
states = [i for i in range(16)]
# 声明状态价值，并初始化各状态价值为0
actions = ["N", "E", "S", "W"]
# 声明行为对状态的改变,move north then the current state -4
ds_actions = {"N": -4, "E": 1, "S": 4, "W": -1}
gramma = 1.00  # 衰减系数
alfa=0.01
value = [0 for x in range(16)]


# determine the next state according to the current state with the action
def nextState(s, a):
    next_state = s
    # 边界情况
    if (s % 4 == 0 and a == "W") or (s < 4 and a == "N") or ((s + 1) % 4 == 0 and a == "E") or \
            (s > 11 and a == "S"):
        pass
    else:
        ds = ds_actions[a]
        next_state = s + ds
    return next_state


# get the reward when leaving the state
def rewardOf(s):
    return 0 if s in [0, 15] else -1


# check whether the terminatestate
def isTerminateState(s):
    return s in [0, 15]


def isNotTerminateState(s):
    return s not in [0, 15]


def line(actions):
    start=random.choice(states)
    current = start
    sample = []
    move = []
    act = random.choice(actions)
    sample.append(current)
    move.append(act)
    while isNotTerminateState(current):
        act = random.choice(actions)
        move.append(act)
        current = nextState(current, act)
        sample.append(current)
    return sample, move





def printValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]), end=" ")
        if (i + 1) % 4 == 0:
            print("")
    print()



def TD(sample):

    for index,s in enumerate(sample):
        #if isTerminateState(s):
           # continue
        if index==0:
            continue
        last=sample[index-1]#last state
        value[last] = value[last] + alfa * (-1 + value[s] - value[last])





def main():


    for i in range(1000000):
        example, moves = line(actions)

        TD(example)
        # print(example)
    printValue(value)


    # printValue(value)



if __name__ == '__main__':
    main()