package strategy;

import java.util.ArrayList;

import neuralNetwork.TrainExample;

public abstract class NNQLearningStrats extends QLearningStrategy{

	public ArrayList<TrainExample> trainExamples = new ArrayList<TrainExample>();
	
	public NNQLearningStrats(double epsilon, double gamma, double alpha) {
		super(epsilon, gamma, alpha);
		// TODO Auto-generated constructor stub
	}


	public abstract void learn(ArrayList<TrainExample> trainExamples);
	
	
	
}
