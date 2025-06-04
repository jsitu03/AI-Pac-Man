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
    
    while len(stack) > 0:
        successor, actions = stack.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for newSr in problem.successorStates(successor):
                if newSr[0] not in visited:
                    stack.push((newSr[0], actions + [newSr[1]]))
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
    
    while len(queue) > 0:
        successor, actions = queue.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for newSr in problem.successorStates(successor):
                if newSr[0] not in visited:
                    queue.push((newSr[0], actions + [newSr[1]]))
    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    pq = PriorityQueue()
    visited = []
    pq.push((start, [], 0), 0)
    
    while len(pq) > 0:
        successor, actions, cost = pq.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for newSr in problem.successorStates(successor):
                if newSr[0] not in visited:
                    newAction = actions + [newSr[1]]
                    newCost = cost + newSr[2]
                    pq.push((newSr[0], newAction, newCost), newCost)
    return []


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    pq = PriorityQueue()
    visited = []
    pq.push((start, [], 0), 0)
    
    while len(pq) > 0:
        successor, actions, cost = pq.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for newSr in problem.successorStates(successor):
                if newSr[0] not in visited:
                    newAction = actions + [newSr[1]]
                    newCost = cost + newSr[2]
                    pq.push((newSr[0], newAction, newCost), newCost + heuristic(newSr[0], problem))
    return []