package main;

import controller.GameController;
import motor.Maze;
import motor.PacmanGame;
import strategy.ApproximateQLearningStrategy;
import strategy.QLearningStrategy;
import strategy.Strategy;
import strategy.TabuLarQLearning;
import view.View;

public class main_debugMode {

	public static void main(String[] args) {
		
		double gamma = 0.95;
		double epsilon = 0.1;
		double alpha = 0.1;
		
		boolean nightmareMode = false;
//		boolean nightmareMode = true;
		
		
	
		
//		String chemin_maze = "src/layout/very_very_smallMaze.lay";
//		String chemin_maze = "src/layout/very_smallMaze.lay";
//		String chemin_maze = "src/layout/mediumClassic.lay";
		String chemin_maze = "src/layout/originalClassic.lay";
//		String chemin_maze = "src/layout/bigMaze.lay";
//		String chemin_maze = "src/layout/xxx.lay";
		PacmanGame _motor = new PacmanGame(chemin_maze, 1000, (long) 100);
		

		
		QLearningStrategy strat = new ApproximateQLearningStrategy(epsilon, gamma, alpha,_motor.getMaze().getSizeX() - 2, _motor.getMaze().getSizeY() - 2);
//		QLearningStrategy strat = new TabuLarQLearning(epsilon, gamma, alpha, _motor.getMaze().getSizeX() - 2, _motor.getMaze().getSizeY() - 2);
		
		strat.setModeTrain(true);
		
		
		_motor.initGameQLearning(strat, nightmareMode);
		
		GameController controller = GameController.getInstance(_motor);
		View _view = View.getInstance(controller, _motor, false);

	}
	
	
}
