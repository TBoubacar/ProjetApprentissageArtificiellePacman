package strategy;

import java.io.Serializable;
import java.util.ArrayList;

import agent.AgentAction;
import agent.PositionAgent;
import motor.PacmanGame;
import neuralNetwork.TrainExample;


public abstract class QLearningStrategy implements Strategy{

	protected double epsilon;
	protected double gamma;
	protected double alpha;
	
	private boolean modeTrain;
	
	
	

	public QLearningStrategy(double epsilon, double gamma, double alpha) {
		
		this.epsilon = epsilon;
		this.gamma = gamma;
		this.alpha = alpha;
		
		
	}
	
	
	public AgentAction play(PacmanGame game, PositionAgent positionAgent, PositionAgent objectif) {
		
		return this.chooseAction(game);
	}
	
	
	
	

	
	public abstract AgentAction chooseAction(PacmanGame state);
	
	
	
	public abstract void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState);
	
	
	public boolean isModeTrain() {
		return modeTrain;
	}


	public void setModeTrain(boolean modeTrain) {
		this.modeTrain = modeTrain;
	}

	
}
