package main;

import java.util.ArrayList;

import controller.GameController;
import motor.Game;
import motor.Maze;
import motor.PacmanGame;
import strategy.ApproximateQLearningStrategy;
import strategy.QLearningStrategy;
import strategy.Strategy;
import strategy.TabuLarQLearning;
import view.View;

public class main_standardMode {

	public static void main(String[] args) {
		
		double gamma = 0.95;
		double epsilon = 0.1;
		double alpha = 0.01;
		boolean nightmareMode = false;
//		boolean nightmareMode = true;
		
		
		
//		String chemin_maze = "src/layout/very_smallMaze.lay";
		String chemin_maze = "src/layout/very_very_smallMaze.lay";
//		String chemin_maze = "src/layout/mediumClassic.lay";

		
	    Maze _maze = null;
	    
		try {
			_maze = new Maze(chemin_maze);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
//		QLearningStrategy strat = new ApproximateQLearningStrategy(epsilon, gamma, alpha);
		
		QLearningStrategy strat = new TabuLarQLearning(epsilon, gamma, alpha, _maze.getSizeX() - 2, _maze.getSizeY() - 2);
		
		
		
		
		//Nombre de simulations séquentielles lancees pour calculer la recompense moyenne en mode train
		int Ntrain = 100;
		
		
		//Nombre de simulations parallèle lancees pour calculer la recompense moyenne en mode test
		int Ntest = 100;
		
		
		//Nombre max de tours d'une partie de pacman
		int maxTurnPacmanGame = 200;
		
		
		
		while(true) {

			strat.setModeTrain(false);
			System.out.println("Visualization mode");
			vizualize(maxTurnPacmanGame, chemin_maze, strat, nightmareMode);
			
			
			//Evaluation du score moyen de la strategie
			strat.setModeTrain(false);
			System.out.println("Eval average score - test mode");
			eval(Ntest, maxTurnPacmanGame, chemin_maze, strat, nightmareMode);
			

			
			
			
			//Joue N simulations du jeu en mode apprentissage
			strat.setModeTrain(true);
			System.out.println("Play and collect examples - train mode");
			learn(Ntrain, maxTurnPacmanGame, chemin_maze, strat, nightmareMode);

		}
		


	}
	
	
	
	

	
	public static void learn(int nbSimulations, int maxTurnPacmanGame, String chemin_maze, QLearningStrategy strat, boolean nightmareMode) {
		

		int globalReward = 0;
		
		for(int i = 0; i < nbSimulations; i++ ) {
			

			PacmanGame _motor = new PacmanGame(chemin_maze, maxTurnPacmanGame, (long) -1);	
			_motor.initGameQLearning(strat, nightmareMode);


			_motor.launch();
			

			try {
				((Game)_motor).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			globalReward += _motor.getScore();
			
		}
		
		
		System.out.println("Average global reward - mode train: " + globalReward/nbSimulations);
		
		
		
//		ArrayList<TrainExample> trainExamples = new ArrayList<TrainExample>();

		
//		for(int i = 0; i < nbSimulations; i++ ) {
//			
//			ArrayList<TrainExample> trainExamplesGame =  pacmanGames.get(i).trainExamples;
//
//			
//			for(int j = 0; j < trainExamplesGame.size(); j++) {
//				
//				trainExamples.add(trainExamplesGame.get(j));
//				
//			}
//			
//		}


		
	}

	
	
	

	public static void eval(int nbSimulations, int maxTurnPacmanGame, String chemin_maze, QLearningStrategy strat, boolean nightmareMode) {
		

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
		
		System.out.println("Average global reward - mode test : " + globalReward/nbSimulations);
		



	}
	
	
	
	private static void vizualize(int maxTurnPacmanGame, String chemin_maze, QLearningStrategy strat, boolean nightmareMode) {
		
		

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
