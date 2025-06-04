import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        foodItems = [distance.manhattan(newPosition, food) for food in foodList]

        minFoodItem = 999999
        if len(foodItems):
            minFoodItem = min(foodItems)

        scaredTime = min(newScaredTimes)

        ghosts = [distance.manhattan(newPosition, ghost) for ghost in ghostPositions]

        closestGhost = min(ghosts)
        if closestGhost == 0:
            closestGhost = 1

        ghostScore = 0
        if scaredTime == 0 and closestGhost < 2:
            ghostScore = - 1000 / closestGhost
        elif scaredTime == 0 and closestGhost >= 2:
            ghostScore = - 2.0 / closestGhost
        else:
            ghostScore = 0.5 / closestGhost

        return successorGameState.getScore() + 1.0 / minFoodItem + ghostScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        def minimax(state, depth, agent):
            if state.isLose() or state.isWin() or depth == 0:
                return self.getEvaluationFunction()(state)
            if agent == 0:
                max_val = -999999
                actions = state.getLegalActions()
                actions.remove(Directions.STOP)
                for a in actions:
                    successor = state.generateSuccessor(agent, a)
                    for ghosts in range(state.getNumAgents() - 1):
                        ev = minimax(successor, depth - 1, ghosts + 1)
                        max_val = max(max_val, ev)
                return max_val
            else:
                min_val = 999999
                actions = state.getLegalActions(agent)
                for a in actions:
                    successor = state.generateSuccessor(agent, a)
                    ev = minimax(successor, depth, 0)
                    min_val = min(min_val, ev)
                return min_val
        actions = state.getLegalActions()
        actions.remove(Directions.STOP)
        depth = self.getTreeDepth()
        best_op = None
        best_val = -999999
        for action in actions:
            successor = state.generateSuccessor(0, action)
            old_val = best_val
            best_val = max(best_val, minimax(successor, depth, 0))
            if old_val != best_val:
                best_op = action
        return best_op

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        legalMoves = state.getLegalActions()
        if not legalMoves:
            return Directions.STOP

        turn = 0
        agents = state.getNumAgents()
        alpha = -999999
        beta = 999999
        bestScore = -999999
        bestAction = Directions.STOP

        for action in legalMoves:
            if action == Directions.STOP:
                continue
            su = state.generatePacmanSuccessor(action)
            score = self.minimax(su, (turn + 1) % agents, self.getTreeDepth() - 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def minimax(self, state, turn, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        if (turn == 0):
            depth -= 1
            return self.maxValue(state, turn, depth, alpha, beta)
        else:
            return self.minValue(state, turn, depth, alpha, beta)

    def maxValue(self, state, turn, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        score = -999999
        legalMoves = state.getLegalActions(turn)
        for action in legalMoves:
            if action == Directions.STOP:
                continue
            su = state.generatePacmanSuccessor(action)
            ms = max(score, self.minimax(su, (turn + 1) % state.getNumAgents(), depth, alpha, beta))
            score = ms
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return score

    def minValue(self, state, turn, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)
        score = 999999
        legalMoves = state.getLegalActions(turn)
        for action in legalMoves:
            if action == Directions.STOP:
                continue
            su = state.generateSuccessor(turn, action)
            ms = min(score, self.minimax(su, (turn + 1) % state.getNumAgents(), depth, alpha, beta))
            score = ms
            if score <= alpha:
                return score
            if score < beta:
                beta = score
        return score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):
        numAgents = gamestate.getNumAgents()

        def expectimax(self, state, agent, depth, prevAction):
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prevAction)
            
            def maxValue(self, state, agent, depth):
                maximum = -999999
                returnAction = Directions.STOP
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    cost, move = expectimax(self, succ, agent + 1, depth, action)
                    maximum = max(maximum, cost)
                    if maximum == cost:
                        returnAction = action
                return (maximum, returnAction)

            def chanceValue(self, state, agent, depth):
                returnAction = Directions.STOP
                actions = []
                costs = []
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    actions.append(action)
                    if agent + 1 == numAgents:
                        cost, move = expectimax(self, succ, 0, depth + 1, action)
                        costs.append(cost)
                    else:
                        cost, move = expectimax(self, succ, agent + 1, depth, action)
                        costs.append(cost)
                expectValue = sum(costs) / len(costs)
                returnAction = random.choice(actions)
                return (expectValue, returnAction)

            if agent == 0:
                return maxValue(self, state, agent, depth)
            else:
                return chanceValue(self, state, agent, depth)

        evaluation, action = expectimax(self, gamestate, 0, 0, Directions.STOP)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
