from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
from pacai.core.actions import Actions


def createTeam(firstIndex, secondIndex, isRed,
        first = 'OffensiveAgent',
        second = 'DefensiveAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indices.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffensiveAgent(firstIndex),
        DefensiveAgent(secondIndex),
    ]

from pacai.util import priorityQueue

def aStarSearch(gameState, start, goal, agentIndex, heuristic):
    """
    A* Search implementation to compute the shortest path to the goal.
    """
    walls = gameState.getWalls()
    pq = priorityQueue.PriorityQueue()
    pq.push((start, [], 0), 0)  # (current position, actions, current cost)

    visited = set()

    while not pq.isEmpty():
        currentPosition, actions, currentCost = pq.pop()

        if currentPosition in visited:
            continue
        visited.add(currentPosition)

        # Check if we reached the goal
        if currentPosition == goal:
            return actions  # Return the list of actions

        # Expand neighbors
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            nextPosition = (int(currentPosition[0] + dx), int(currentPosition[1] + dy))

            if not walls[nextPosition[0]][nextPosition[1]]:  # Check if the next position is valid
                newCost = currentCost + 1
                heuristicCost = heuristic(nextPosition, goal)
                pq.push((nextPosition, actions + [direction], newCost), newCost + heuristicCost)

    return []  # Return empty list if no path is found

def manhattanHeuristic(position, goal):
    """
    Manhattan distance heuristic for A* Search.
    """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

import random
class QLearningCaptureAgent(CaptureAgent):
    def __init__(self, index, alpha=0.5, gamma=0.9, epsilon=0.1):
        super().__init__(index)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.qValues = {}  # Q-Value table
    
    def getQValue(self, state, action):
        """
        Get the Q-value for a given state and action.
        Ensure state is converted to a hashable tuple.
        """
        if isinstance(state, dict):  # If already processed as a dictionary
            hashableState = tuple(state.items())
        else:
            hashableState = state  # Assume state is already processed as hashable

        return self.qValues.get((hashableState, action), 0.0)

    
    def computeActionFromQValues(self, gameState):
        """
        Compute the best action based on Q-values for a given state.
        """
        stateFeatures = self.getStateFeatures(gameState)  # Extract features from gameState
        hashableState = tuple(stateFeatures.items())  # Convert features to a hashable tuple

        # Fetch legal actions from the gameState
        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return None

        maxQValue = float('-inf')
        bestActions = []
        for action in legalActions:
            qValue = self.getQValue(hashableState, action)  # Use hashableState here
            if qValue > maxQValue:
                maxQValue = qValue
                bestActions = [action]
            elif qValue == maxQValue:
                bestActions.append(action)

        return random.choice(bestActions) if bestActions else None
    

    def getBoundaryPosition(self, gameState):
        """
        Get a list of positions on the boundary that the agent can move to.
        """
        walls = gameState.getWalls()
        layoutWidth = walls._width
        boundaryX = layoutWidth // 2
        if not self.red:
            boundaryX -= 1

        # Collect all accessible boundary positions
        boundaryPositions = [
            (boundaryX, y)
            for y in range(walls.height)
            if not walls[boundaryX][y]
        ]

        # Return the closest boundary position based on the agent's current location
        myPos = gameState.getAgentState(self.index).getPosition()
        return min(boundaryPositions, key=lambda pos: self.getMazeDistance(myPos, pos))


    def chooseAction(self, gameState):
        """
        Choose an action based on Q-values and game state.
        """
        # Use the original gameState to get legal actions
        actions = gameState.getLegalActions(self.index)
        
        # Simplified state for Q-learning (feature extraction)
        simplifiedState = self.getStateFeatures(gameState)

        # Use epsilon-greedy policy for action selection
        if random.random() < self.epsilon:
            # Exploration: choose a random legal action
            chosenAction = random.choice(actions)
        else:
            # Exploitation: choose the best action based on Q-values
            maxQValue = float('-inf')
            chosenAction = None
            for action in actions:
                qValue = self.getQValue(simplifiedState, action)
                if qValue > maxQValue:
                    maxQValue = qValue
                    chosenAction = action

        # Update Q-values based on the chosen action
        successor = self.getSuccessor(gameState, chosenAction)
        nextSimplifiedState = self.getStateFeatures(successor)
        reward = self.getReward(gameState, successor)
        self.update(simplifiedState, chosenAction, nextSimplifiedState, reward)

        return chosenAction


    def update(self, state, action, nextState, reward):
        """
        Update Q-values using the Q-learning update rule.
        """
        if not isinstance(state, dict):
            state = self.getStateFeatures(state)  # Extract features if raw state passed

        if not isinstance(nextState, dict):
            nextState = self.getStateFeatures(nextState)  # Extract features if raw state passed

        hashableState = tuple(state.items())
        hashableNextState = tuple(nextState.items())

        oldQValue = self.getQValue(hashableState, action)
        nextValue = self.computeValueFromQValues(nextState)
        self.qValues[(hashableState, action)] = (
            (1 - self.alpha) * oldQValue + self.alpha * (reward + self.gamma * nextValue)
        )

    
    def getStateFeatures(self, gameState):
        """
        Extract relevant features for Q-learning from the game state.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(gameState).asList()
        foodDist = min([self.getMazeDistance(myPos, food) for food in foodList]) if foodList else 0
        capsules = self.getCapsules(gameState)

        return {
            'foodCarried': self.foodCarried,
            'foodDist': foodDist,
            'capsules': len(capsules),
            'isHome': self.isInHomeTerritory(myPos, gameState),
        }


"""
import random
from pacai.util import counter
import time
import logging
from pacai.util import util

class OffensiveAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):


        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)

        values = [self.evalFcn(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evalFcn(self, state, action):
        weights = self.weightsGenerator(state, action)
        features = self.featureGenerator(state, action)
        return features * weights

    def featureGenerator(self, state, action):

        successor = self.getSuccessor(state, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        features = counter.Counter()

        features['successorScore'] = self.getScore(successor)

        foodList = self.getFood(successor).asList()
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        guards = [a for a in enemies if a.isGhost() and a.getPosition() is not None]

        guardDists = [self.getMazeDistance(myPos, a.getPosition()) for a in guards]
        if (len(guardDists) > 0):
            minGuardDistance = min(guardDists)
            features['guardDistance'] = minGuardDistance

        capsuleList = self.getCapsules(successor)
        if (len(capsuleList) > 0):
            minCapsuleDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
            features['distanceToCapsule'] = minCapsuleDistance

        allies = [successor.getAgentState(i) for i in self.getTeam(successor)]
        if (len(allies) > 0):
            allyDists = [self.getMazeDistance(myPos, a.getPosition()) for a in allies]
            features['allyDistance'] = min(allyDists)
        return features

    def weightsGenerator(self, state, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'guardDistance': 0.75,
            'allyDistance': 0,
            'distanceToCapsule': -0.2
        }

    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
"""

class OffensiveAgent(CaptureAgent):
    """
    A strategic offensive agent that collects food, avoids ghosts, and returns home efficiently.
    """

    def __init__(self, index):
        super().__init__(index)
        self.foodCarried = 0  # Tracks food collected by the agent

    def chooseAction(self, gameState):
        """
        Choose the best action based on the evaluation function.
        """
        # Get all legal actions
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        maxEval = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            evalScore = self.evaluate(successor)
            if evalScore > maxEval:
                maxEval = evalScore
                bestAction = action

        return bestAction

    
    def getSuccessor(self, gameState, action):
        """
        Returns the successor game state after the given action is taken.
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor
        """
        Check if the agent is in its home territory.
        """
        walls = gameState.getWalls()
        layoutWidth = walls._width
        boundaryX = layoutWidth // 2
        if not self.red:  # Adjust for blue team
            boundaryX -= 1

        if self.red:
            return position[0] < boundaryX
        else:
            return position[0] > boundaryX

    def evaluate(self, successor):
        """
        Evaluate the successor state for food, capsules, and opponent proximity.
        """
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 1. Food distance (minimize distance to the nearest food)
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            foodDist = min([self.getMazeDistance(myPos, food) for food in foodList])
        else:
            foodDist = 0

        # 2. Capsule distance (minimize distance to the nearest capsule)
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            capsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
        else:
            capsuleDist = 100  # No capsules nearby

        # 3. Ghost avoidance (penalize being too close to ghosts)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        if len(ghosts) > 0:
            ghostDistances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            ghostDist = min(ghostDistances)
            if ghostDist <= 2:
                ghostPenalty = -200  # Strong penalty for being close
            elif ghostDist <= 5:
                ghostPenalty = -50  # Moderate penalty for being nearby
            else:
                ghostPenalty = 0
        else:
            ghostPenalty = 0

        # 5. Encourage capsule consumption when ghosts are close
        if len(capsules) > 0 and ghostPenalty < -50:  # Ghosts are nearby
            capsuleIncentive = 300 - 5 * capsuleDist  # Strongly incentivize eating capsules
        else:
            capsuleIncentive = 0

        # Weighted evaluation score
        evalScore = (
            -2 * foodDist  # Prioritize collecting food
            -3 * capsuleDist  # Incentivize capsules
            + ghostPenalty  # Penalize proximity to ghosts
            + capsuleIncentive  # Strong incentive for capsule consumption
        )

        return evalScore


"""
class OffensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.foodCarried = 0  # Tracks food collected by the agent
    
    def chooseAction(self, gameState):
        
        #Choose an action using Q-Learning for high-level decision-making and A* for execution.
        
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP

        # High-level action decision
        highLevelAction = self.computeActionFromQValues(gameState)  # Pass gameState here

        # Execute the high-level action using A* Search or other strategies
        myPos = gameState.getAgentState(self.index).getPosition()
        if highLevelAction == "collectFood":
            foodList = self.getFood(gameState).asList()
            if foodList:
                closestFood = min(foodList, key=lambda food: self.getMazeDistance(myPos, food))
                path = aStarSearch(gameState, myPos, closestFood, self.index, manhattanHeuristic)
                return path[0] if path else Directions.STOP

        elif highLevelAction == "returnHome":
            homeBoundary = self.getBoundaryPosition(gameState)
            path = aStarSearch(gameState, myPos, homeBoundary, self.index, manhattanHeuristic)
            return path[0] if path else Directions.STOP

        elif highLevelAction == "eatCapsule":
            capsules = self.getCapsules(gameState)
            if capsules:
                closestCapsule = min(capsules, key=lambda cap: self.getMazeDistance(myPos, cap))
                path = aStarSearch(gameState, myPos, closestCapsule, self.index, manhattanHeuristic)
                return path[0] if path else Directions.STOP

        # Default to a random legal action if no high-level goal is met
        return random.choice(actions)

    
    def getStateFeatures(self, gameState):
        #Extract a simplified representation of the gameState for Q-Learning.
     
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(gameState).asList()
        foodDist = min([self.getMazeDistance(myPos, food) for food in foodList]) if foodList else 0
        capsules = self.getCapsules(gameState)

        return {
            'foodCarried': self.foodCarried,
            'foodDist': foodDist,
            'capsules': len(capsules),
            'isHome': self.isInHomeTerritory(myPos, gameState),
        }

    def isInHomeTerritory(self, position, gameState):
       # Determine if the agent is in its home territory.
        walls = gameState.getWalls()
        layoutWidth = walls._width
        boundaryX = layoutWidth // 2
        if not self.red:
            boundaryX -= 1

        if self.red:
            return position[0] < boundaryX
        else:
            return position[0] > boundaryX
"""


class DefensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Defensive agent decision-making using Q-Learning and A*.
        """
        # Identify invaders
        invaders = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i).isPacman() and gameState.getAgentState(i).getPosition() is not None
        ]

        myPos = gameState.getAgentState(self.index).getPosition()

        if invaders:
            # Chase the closest invader
            closestInvader = min(invaders, key=lambda inv: self.getMazeDistance(myPos, inv.getPosition()))
            path = aStarSearch(gameState, myPos, closestInvader.getPosition(), self.index, manhattanHeuristic)
            return path[0] if path else Directions.STOP
        else:
            # Patrol critical food or boundary
            criticalFood = self.getCriticalFood(gameState)
            if criticalFood:
                path = aStarSearch(gameState, myPos, criticalFood, self.index, manhattanHeuristic)
                return path[0] if path else Directions.STOP

            # Patrol the boundary as a fallback
            patrolPos = self.getBoundaryPosition(gameState)
            path = aStarSearch(gameState, myPos, patrolPos, self.index, manhattanHeuristic)
            return path[0] if path else Directions.STOP


    def getCriticalFood(self, gameState):
        """
        Find the most critical food to defend (e.g., closest to the boundary).
        """
        foodDefending = self.getFoodYouAreDefending(gameState).asList()
        boundaryX = gameState.getWalls()._width // 2
        if not self.red:
            boundaryX -= 1

        if foodDefending:
            return min(foodDefending, key=lambda food: abs(food[0] - boundaryX))
        return None
