package main;

import java.util.ArrayList;

import controller.GameController;
import motor.Game;
import motor.PacmanGame;
import neuralNetwork.TrainExample;
import strategy.QLearningStrategy;
import view.View;
import strategy.ApproximateQLearningStrategy;
import strategy.ApproximateQLearningStrategy_withNN;
import strategy.DeepQLearningStrategy;
import strategy.NNQLearningStrats;


public class main_batchMode {

	public static void main(String[] args) {

		boolean nightmareMode = true;
		
		double gamma = 0.95;
		double epsilon = 0.1;
		double alpha = 0.001;
		

		
		NNQLearningStrats strat = new ApproximateQLearningStrategy_withNN(epsilon, gamma, alpha, 10, 100);
//		NNQLearningStrats strat  = new DeepQLearningStrategy(epsilon, gamma, alpha, 5, 10, 100);
			
		
		
		String chemin_maze = "src/layout/originalClassic.lay";
		
		//Nombre de simulations lancees pour calculer la recompense moyenne et collecter des exemples
		int N = 100;
		
		//Nombre max de tours d'une partie de pacman
		int maxTurnPacmanGame = 300;
		
		
		
		while(true) {
			
			//Joue N simulation du jeu et collecte les exemples d'entrainement
			
			strat.setModeTrain(true);
			System.out.println("Play and collect examples - train mode");
			ArrayList<TrainExample> trainExamples = play(N, maxTurnPacmanGame, chemin_maze, strat, nightmareMode);
			
			//Apprend a partir des exemples d'entrainement
			System.out.println("Learn model");
			strat.learn(trainExamples);
			strat.trainExamples.clear();
			
			
			//Evaluation du score moyen de la strategie
			strat.setModeTrain(false);
			System.out.println("Eval average score - test mode");
			play(N, maxTurnPacmanGame, chemin_maze, strat, nightmareMode);
			
			System.out.println("Visualization mode");
			vizualize(maxTurnPacmanGame, chemin_maze, strat, nightmareMode);
			
		}
		
	}
	
	



	public static ArrayList<TrainExample> play(int nbSimulations, int maxTurnPacmanGame, String chemin_maze, NNQLearningStrats strat, boolean nightmareMode) {
		

		ArrayList<PacmanGame> pacmanGames = new ArrayList<PacmanGame>();
		
		for(int i = 0; i < nbSimulations; i++ ) {
			PacmanGame _motor = new PacmanGame(chemin_maze, maxTurnPacmanGame, (long) -1);	
			_motor.initGameQLearning(strat, nightmareMode);
			pacmanGames.add(_motor);
		}

		System.out.println("Launch " + nbSimulations + " pacman games");
		
		for(int i = 0; i < nbSimulations; i++ ) {
			
			pacmanGames.get(i).launch();
		}
		
		for(int i = 0; i < nbSimulations; i++ ) {
		
			try {
				((Game)pacmanGames.get(i)).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		int globalReward = 0;
		
		for(int i = 0; i < nbSimulations; i++ ) {
			globalReward += pacmanGames.get(i).getScore();
		}
		
		System.out.println("Average global reward : " + globalReward/nbSimulations);
		
		
		
		ArrayList<TrainExample> trainExamples = strat.trainExamples;
		
		System.out.println("total number examples : " + trainExamples.size());
		
		
		return trainExamples;
		
	}

	
	private static void vizualize(int maxTurnPacmanGame, String chemin_maze, NNQLearningStrats strat, boolean nightmareMode) {
		
		

		PacmanGame _motor = new PacmanGame(chemin_maze, maxTurnPacmanGame, (long) 100);
		GameController controller = GameController.getInstance(_motor);
		View _view = View.getInstance(controller, _motor, false);
		
		_motor.initGameQLearning(strat, nightmareMode);
		

		_view.btnRun.setEnabled(false);
		_view.btnPause.setEnabled(true);
		
		controller._motor = _motor;
		
		_view._motor = _motor;
		_view.btnRun.setEnabled(false);
		_view.btnPause.setEnabled(true);
		_motor.addObserver(_view);
		
		
		_motor.launch();
		
		try {
			((Game)_motor).join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		

	}


}
