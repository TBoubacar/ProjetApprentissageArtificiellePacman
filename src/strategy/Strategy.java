package strategy;

import java.util.ArrayList;

import agent.AgentAction;
import agent.PositionAgent;
import agent.typeAgent;
import motor.Maze;
import motor.PacmanGame;


public interface Strategy {
	

	
	public AgentAction play(PacmanGame state, PositionAgent positionAgent, PositionAgent objectif);
	

	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState);




}
