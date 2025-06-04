"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    start = problem.startingState()
    stack = Stack()
    visited = []
    stack.push((start, []))
    if problem.isGoal(start):
        return []
    
    while not stack.isEmpty():
        successor, actions = stack.pop()
        if problem.isGoal(successor):
            return actions
        for child, dir in problem.successorStates(actions):
            if child not in visited:
                stack.push((child, actions + [dir]))
                visited.appened(child)
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    start = problem.startingState()
    queue = Queue()
    visited = []
    queue.push((start, []))
    if problem.isGoal(start):
        return []
    
    while not queue.isEmpty():
        successor, actions = queue.pop()
        if problem.isGoal(successor):
            return actions
        for child, dir in problem.successorStates(actions):
            if child not in visited:
                queue.push((child, actions + [dir]))
                visited.appened(child)
    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    pq = PriorityQueue()
    visited = []
    pq.push((start, []), 0)
    if problem.isGoal(start):
        return []
    
    while not pq.isEmpty():
        successor, actions = pq.pop()
        if problem.isGoal(successor):
            return actions
        for child, dir in problem.successorStates(actions):
            if child not in visited:
                childActions = actions + [dir]
                childCost = problem.actionCost(childActions)
                pq.push((child, childActions), childCost)
                visited.appened(child)
    return []


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    pq = PriorityQueue()
    visited = []
    pq.push((start, []))
    if problem.isGoal(start):
        return []
    
    while not pq.isEmpty():
        successor, actions = pq.pop()
        if problem.isGoal(successor):
            return actions
        for child, dir in problem.successorStates(actions):
            if child not in visited:
                childActions = actions + [dir]
                childCost = problem.actionCost(childActions)
                pq.push((child, childActions), childCost + heuristic(child, problem))
                visited.appened(child)
    return []

