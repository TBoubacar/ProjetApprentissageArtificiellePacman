package strategy;

import java.util.ArrayList;

import agent.AgentAction;

import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;
import neuralNetwork.TrainExample;


import java.util.Random;


public class ApproximateQLearningStrategy_withNN extends NNQLearningStrats{
	private final int nbAction = 4;
	private NeuralNetWorkDL4J nn;
	private double baseEpsilon;
	private int batchSize;
	private int sizeX;
	private int sizeY;
	private int nEpochs;

	
	public ApproximateQLearningStrategy_withNN(double epsilon, double gamma, double alpha,  int nEpochs, int batchSize) {
		// TODO Auto-generated constructor stub
		super(epsilon, gamma, alpha);
		this.nn = new NeuralNetWorkDL4J(alpha, 0, this.nbAction+1, 1 );
		this.sizeX = 11;
		this.sizeY = 11;
		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
		this.setBaseEpsilon(epsilon);
	}
	
	@Override
	public AgentAction chooseAction(PacmanGame state) {
		// TODO Auto-generated method stub
		setSizeX(state.getMaze().getSizeX()-2);
		setSizeY(state.getMaze().getSizeY()-2);
		AgentAction actionChoose; 

		Random random = new Random();
		if(Math.random() < epsilon) {			
			do {
				int n = (int) (Math.random() * (this.nbAction));
				actionChoose = new AgentAction(n);				
			} while(! state.isLegalMove(state.pacman, actionChoose));
			
		} else {
			double maxQvalue = -999;
			actionChoose = this.chooseAleatoireAction(state);
			
			for(int a=0; a < this.nbAction; a++) {		
				double[] features = extractFeatures(state, new AgentAction(a));
				double qValue = this.nn.predict(features)[0];
								
				if(qValue > maxQvalue) {
					maxQvalue = qValue;
					actionChoose = new AgentAction(a);
				} else if(qValue  == maxQvalue) {
					if(random.nextBoolean()) {
						maxQvalue = qValue;
						actionChoose = new AgentAction(a);
					}
				}
				
			}
			return actionChoose;
		}
		return actionChoose;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		// TODO Auto-generated method stub
		double maxQvalue_nextState = -999;
		
		for(int a=0; a < this.nbAction; a++) {
			double[] features = extractFeatures(nextState, new AgentAction(a));
			double qValue_nextState = this.nn.predict(features)[0];
			if(qValue_nextState > maxQvalue_nextState) {
				maxQvalue_nextState = qValue_nextState;
			}
		}
		
		double[] targetQ = new double[1];
		
		targetQ[0] = reward + gamma*maxQvalue_nextState;
		
		
		double[] currentFeatures = extractFeatures(state, action);
		
		TrainExample trainExample = new TrainExample(currentFeatures, targetQ);
		this.trainExamples.add(trainExample);
	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// TODO Auto-generated method stub
		this.nn.fit(trainExamples, this.nEpochs, this.batchSize, this.baseEpsilon);
	}

	
	public double scalarProduct(double[] weights, double[] features) {
		double res = 0;
		for(int i = 0; i < weights.length; i++) {
			res += weights[i]*features[i];
		}
		return res;
	}
	
	public double[] extractFeatures(PacmanGame state, AgentAction action ) {
		double[] features = new double[this.nbAction+1];
		features[0] = 1;
		this.rempliCaseUnFeatures(state, action, features);
		this.rempliCaseDeuxFeatures(state, action, features);
		this.rempliCaseTroisFeatures(state, action, features);
		this.rempliCaseQuatreFeatures(state, action, features);
		return features;
	}

	
	/*---		AJOUT DE CODE PLUS BAS		---*/
	public int calculCoutByRadarPacgumInLineAction(PacmanGame state, AgentAction action) {
		// TODO Auto-generated method stub
		if ( action.get_direction() == AgentAction.NORTH) {
			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
				if (state.getMaze().isWall(state.pacman.get_position().getX(), i)) return -1;
				if (this.myLegalMove(state.pacman.get_position().getX(), i) && state.getMaze().isFood(state.pacman.get_position().getX(), i) || state.getMaze().isCapsule(state.pacman.get_position().getX(), i) && ! state.getMaze().isWall(state.pacman.get_position().getX(), i))
					return 1;
			}
		} else if ( action.get_direction() == AgentAction.SOUTH) {
			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeY; i++) {
				if (state.getMaze().isWall(state.pacman.get_position().getX(), i)) return -1;
				if (this.myLegalMove(state.pacman.get_position().getX(), i) && state.getMaze().isFood(state.pacman.get_position().getX(), i) || state.getMaze().isCapsule(state.pacman.get_position().getX(), i) && ! state.getMaze().isWall(state.pacman.get_position().getX(), i))
					return 1;
			}
		} else if ( action.get_direction() == AgentAction.EAST) {
			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
				if (state.getMaze().isWall(i, state.pacman.get_position().getY())) return -1;
				if (this.myLegalMove(i, state.pacman.get_position().getY()) && state.getMaze().isFood(i, state.pacman.get_position().getY()) || state.getMaze().isCapsule(i, state.pacman.get_position().getY()) && ! state.getMaze().isWall(i, state.pacman.get_position().getY()))
					return 1;
			}
		} else if ( action.get_direction() == AgentAction.WEST) {
			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
				if (state.getMaze().isWall(i, state.pacman.get_position().getY())) return -1;
				if (this.myLegalMove(i, state.pacman.get_position().getY()) && state.getMaze().isFood(i, state.pacman.get_position().getY()) || state.getMaze().isCapsule(i, state.pacman.get_position().getY()) && ! state.getMaze().isWall(i, state.pacman.get_position().getY()))
					return 1;
			}		
		}
		return 0;

	}
	
	public int calculCoutByRadarPacgum(PacmanGame state, AgentAction action) {
		// TODO Auto-generated method stub
		int xPacman = state.pacman.get_position().getX();
		int yPacman = state.pacman.get_position().getY();
		int nbCoupPossible = 0;
		int nbCoupPossibleX = this.sizeX-xPacman;
		int nbCoupPossibleY = this.sizeY-yPacman;
		
		if (nbCoupPossibleX > nbCoupPossibleY) {
			nbCoupPossible = nbCoupPossibleX;
			if (nbCoupPossibleX > this.sizeX)
				nbCoupPossible = this.sizeX;
		}
		else {
			nbCoupPossible = nbCoupPossibleY;
			if (nbCoupPossibleY > this.sizeY)
				nbCoupPossibleY = this.sizeY;
		}
		
		for(int i=1; i<=nbCoupPossible; ++i) {
			if ((this.canEatPacGum(state, action, xPacman, yPacman-i) &&  ! state.getMaze().isWall( xPacman, yPacman-i)) || (this.canEatPacGum(state, action, xPacman, yPacman+i) && ! state.getMaze().isWall( xPacman, yPacman+i))
			|| (this.canEatPacGum(state, action, xPacman-i, yPacman) && ! state.getMaze().isWall( xPacman-i, yPacman)) || (this.canEatPacGum(state, action, xPacman+i, yPacman)) && ! state.getMaze().isWall( xPacman+i, yPacman)) {
				return i;
			}
		}
		return nbCoupPossible;

	}
	
	private void rempliCaseUnFeatures(PacmanGame state, AgentAction action, double[] features) {
		if ( action.get_direction() == AgentAction.NORTH && this.canEatPacGum(state, action, state.pacman.get_position().getX(), state.pacman.get_position().getY()-1)) {
			features[1] = 1;
			if (this.calculCoutByRadarPacgumInLineAction(state, action)>4)
				features[1] = this.calculCoutByRadarPacgumInLineAction(state, action);
		} else if ( action.get_direction() == AgentAction.SOUTH && this.canEatPacGum(state, action, state.pacman.get_position().getX(), state.pacman.get_position().getY()+1)) {
			features[1] = 1;
			if (this.calculCoutByRadarPacgumInLineAction(state, action)>4)
				features[1] = this.calculCoutByRadarPacgumInLineAction(state, action);
		} else if ( action.get_direction() == AgentAction.EAST && this.canEatPacGum(state, action, state.pacman.get_position().getX()+1, state.pacman.get_position().getY())) {
			features[1] = 1;
			if (this.calculCoutByRadarPacgumInLineAction(state, action)>4)
				features[1] = this.calculCoutByRadarPacgumInLineAction(state, action);
		} else if ( action.get_direction() == AgentAction.WEST && this.canEatPacGum(state, action, state.pacman.get_position().getX()-1, state.pacman.get_position().getY())) {
			features[1] = 1;
			if (this.calculCoutByRadarPacgumInLineAction(state, action)>4)
				features[1] = this.calculCoutByRadarPacgumInLineAction(state, action);
		} else {
			features[1] = 0;
		}
	}
	
	private void rempliCaseDeuxFeatures(PacmanGame state, AgentAction action, double[] features) {
//		features[2] = this.foodPointInDirectionOfPacman(state, action);				// UN PEU BON
		features[2] = this.calculCoutByRadarPacgum(state, action);
	}

	private void rempliCaseTroisFeatures(PacmanGame state, AgentAction action, double[] features) {
		features[3] = this.radarOfPacman(state, action);
	}
	
	private void rempliCaseQuatreFeatures(PacmanGame state, AgentAction action, double[] features) {
//		features[4] = this.distanceMinInSameDirectionPacmanFantom(state, action);
		features[4] = (this.isWayInFrontOfPacman(state, action)) ? 1:-1;
	}	

	private ArrayList<Integer>  isLegalMoveTabId(PacmanGame state) {
		// TODO Auto-generated method stub
		ArrayList<Integer> tabIdLegalMove = new ArrayList<Integer>();
		AgentAction action;
		for (int i = 0; i < this.nbAction; i++) {	// ON REGARDE SI L'ACTION EST POSSIBLE ON LE MET DANS LE TABLEAU SINON ON PASSE
			action = new AgentAction(i);
			if (state.isLegalMove(state.pacman, action)) {
				tabIdLegalMove.add(i);
			}
		}
		if (tabIdLegalMove.size() == 0) return null;
		return tabIdLegalMove;
	}
	
	private AgentAction chooseAleatoireAction(PacmanGame state) {
		// TODO Auto-generated method stub
		ArrayList<Integer> aleatoireAction = this.isLegalMoveTabId(state);
		if (aleatoireAction != null) {
			int idAleatoireAction = aleatoireAction.get(new Random().nextInt(aleatoireAction.size()));
			return new AgentAction(idAleatoireAction);
		}
		return new AgentAction(new Random().nextInt(this.nbAction));
	}
	
	private boolean myLegalMove(int x, int y) {
		// TODO Auto-generated method stub
		return x <= this.sizeX && x >= 1 && y <= this.sizeY && y >= 1; 
	}
	
	private boolean canEatPacGum(PacmanGame state, AgentAction action, int x, int y) {
		if (this.myLegalMove(x, y)) {
			return state.isLegalMove(state.pacman, action) && state.getMaze().isFood(x, y) || state.getMaze().isCapsule(x, y);
		}
		return false;
	}
	
	private int radarOfPacman(PacmanGame state, AgentAction action) {
		// TODO Auto-generated method stub
		int nbpoint = 0;
		if (this.isFoodInDirectionOfPacman(state, action) && this.isWayInFrontOfPacman(state, action)) {
			++nbpoint;
		}
		if (this.isCapsuleInDirectionOfPacman(state, action) && this.isWayInFrontOfPacman(state, action)) {
			nbpoint += 3;
		}
		if (this.isPacmanInDirectionOfFantom(state, action)) {
			nbpoint -= 2;
		}
		if (this.isWayInFrontOfPacman(state, action)) {
			nbpoint += 0.5;
		}
		return nbpoint;
	}

	private boolean isWayInFrontOfPacman(PacmanGame state, AgentAction action) {
		if ( action.get_direction() == AgentAction.NORTH && !state.getMaze().isWall(state.pacman.get_position().getX(), state.pacman.get_position().getY()-1)) {
			return true;
		} else if ( action.get_direction() == AgentAction.SOUTH && ! state.getMaze().isWall(state.pacman.get_position().getX(), state.pacman.get_position().getY()+1)) {
			return true;
		} else if ( action.get_direction() == AgentAction.EAST && ! state.getMaze().isWall(state.pacman.get_position().getX()+1, state.pacman.get_position().getY())) {
			return true;
		} else if ( action.get_direction() == AgentAction.WEST && ! state.getMaze().isWall(state.pacman.get_position().getX()-1, state.pacman.get_position().getY())) {
			return true;
		} else {
			return false;
		}
	}
	private boolean isFoodInDirectionOfPacman(PacmanGame state, AgentAction action) {
		if ( action.get_direction() == AgentAction.NORTH) {
			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
				if (state.getMaze().isFood(state.pacman.get_position().getX(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.SOUTH) {
			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeY; i++) {
				if (state.getMaze().isFood(state.pacman.get_position().getX(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.EAST) {
			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
				if (state.getMaze().isFood(i, state.pacman.get_position().getY()))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.WEST) {
			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
				if (state.getMaze().isFood(i, state.pacman.get_position().getY()))
					return true;
			}		
		}
		return false;
	}
	
	private boolean isCapsuleInDirectionOfPacman(PacmanGame state, AgentAction action) {
		if ( action.get_direction() == AgentAction.NORTH) {
			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
				if (state.getMaze().isCapsule(state.pacman.get_position().getY(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.SOUTH) {
			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeY; i++) {
				if (state.getMaze().isCapsule(state.pacman.get_position().getX(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.EAST) {
			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
				if (state.getMaze().isCapsule(i, state.pacman.get_position().getY()))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.WEST) {
			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
				if (state.getMaze().isCapsule(i, state.pacman.get_position().getY()))
					return true;
			}		
		}
		return false;
	}
	
	private boolean isPacmanInDirectionOfFantom(PacmanGame state, AgentAction action) {
		if ( action.get_direction() == AgentAction.NORTH) {
			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
				if (state.isGhostsAt(state.pacman.get_position().getX(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.SOUTH) {
			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeY; i++) {
				if (state.isGhostsAt(state.pacman.get_position().getX(), i))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.EAST) {
			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
				if (state.isGhostsAt(i, state.pacman.get_position().getY()))
					return true;
			}
		} else if ( action.get_direction() == AgentAction.WEST) {
			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
				if (state.isGhostsAt(i, state.pacman.get_position().getY()))
					return true;
			}		
		}
		return false;
	}


//	private double distanceMinInSameDirectionPacmanFantom(PacmanGame state, AgentAction action) {
//		double disMax = 0;
//		if ( action.get_direction() == AgentAction.NORTH) {
//			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
//				if (state.isGhostsAt(state.pacman.get_position().getY(), i) && ! state.getMaze().isWall(state.pacman.get_position().getY(), state.pacman.get_position().getY()-1))
//					return i;
//			}
//		} else if ( action.get_direction() == AgentAction.SOUTH) {
//			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeX; i++) {
//				if (state.isGhostsAt(state.pacman.get_position().getY(), i) && ! state.getMaze().isWall(state.pacman.get_position().getY(), state.pacman.get_position().getY()+1))
//					return i;
//			}
//		} else if ( action.get_direction() == AgentAction.EAST) {
//			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
//				if (state.isGhostsAt(i, state.pacman.get_position().getX()) && ! state.getMaze().isWall(state.pacman.get_position().getX()+1, state.pacman.get_position().getX()))
//					return i;
//			}
//		} else if ( action.get_direction() == AgentAction.WEST) {
//			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
//				if (state.isGhostsAt(i, state.pacman.get_position().getX()) && ! state.getMaze().isWall(state.pacman.get_position().getX()-1, state.pacman.get_position().getX()))
//					return i;
//			}		
//		}
//		return disMax;
//	}
//	
//	private int foodPointInDirectionOfPacman(PacmanGame state, AgentAction action) {
//		int compteur = 0;
//		if ( action.get_direction() == AgentAction.NORTH) {
//			for (int i = state.pacman.get_position().getY()-1; i > 0; i--) {
//				if (state.getMaze().isFood(state.pacman.get_position().getX(), i))
//					++compteur;
//			}
//		} else if ( action.get_direction() == AgentAction.SOUTH) {
//			for (int i = state.pacman.get_position().getY()+1; i <= this.sizeY; i++) {
//				if (state.getMaze().isFood(state.pacman.get_position().getX(), i))
//					++compteur;
//			}
//		} else if ( action.get_direction() == AgentAction.EAST) {
//			for (int i = state.pacman.get_position().getX()+1; i <= this.sizeX; i++) {
//				if (state.getMaze().isFood(i, state.pacman.get_position().getY()))
//					++compteur;
//			}
//		} else if ( action.get_direction() == AgentAction.WEST) {
//			for (int i = state.pacman.get_position().getX()-1; i > 0; i--) {
//				if (state.getMaze().isFood(i, state.pacman.get_position().getY()))
//					++compteur;
//			}		
//		}
//		return compteur;
//	}
//	
//	private double distanceMinPacmanFantom(PacmanGame state) {
//		double disMin = this.sizeX*this.sizeY;
//		double disEntreFP = 0;
//		
//		for(Agent fantome: state.get_agentsFantom()) {
//			disEntreFP = Math.sqrt(Math.pow(state.pacman.get_position().getX() - fantome.get_position().getX(), 2) + Math.pow(state.pacman.get_position().getY() - fantome.get_position().getY(), 2));
//			if (disMin > disEntreFP) {
//				disMin = disEntreFP;
//			}
//		}
//		return disMin;
//	}
//
//	private int thereFantomeEnVueInRayon(PacmanGame state, int rayon) {
//		int nbFantomVue = 0;
//		int finX = state.pacman.get_position().getX() + rayon;
//		int debutX = state.pacman.get_position().getX()- rayon;		
//		int finY = state.pacman.get_position().getY() + rayon;
//		int debutY = state.pacman.get_position().getY()- rayon;
//		
//		if (finX > this.sizeX) {
//			finX = this.sizeX;
//		} else if (finY > this.sizeY) {
//			finY = this.sizeY;
//		} else if (debutX < 0) {
//			debutX = 0;
//		} else if (debutY < 0) {
//			debutY = 0;
//		}
//		
//		for(Agent fantome: state.get_agentsFantom()) {
//			if (state.pacman.get_position().getX() < fantome.get_position().getX() && fantome.get_position().getX() <= finX &&
//				state.pacman.get_position().getY() < fantome.get_position().getY() && fantome.get_position().getY() <= finY)
//				++nbFantomVue;
//			else if (state.pacman.get_position().getX() < fantome.get_position().getX() && fantome.get_position().getX() <= finX &&
//					state.pacman.get_position().getY() > fantome.get_position().getY() && fantome.get_position().getY() >= debutY)
//				++nbFantomVue;
//			else if (state.pacman.get_position().getX() > fantome.get_position().getX() && fantome.get_position().getX() >= debutX &&
//					state.pacman.get_position().getY() < fantome.get_position().getY() && fantome.get_position().getY() <= finY)
//				++nbFantomVue;
//			else if (state.pacman.get_position().getX() > fantome.get_position().getX() && fantome.get_position().getX() <= finX &&
//					state.pacman.get_position().getY() > fantome.get_position().getY() && fantome.get_position().getY() >= debutY)
//				++nbFantomVue;
//		}
//		
//		return nbFantomVue;
//	}

	/*------	GETTERS AND SETTERS		------*/

	public int getSizeX() {
		return sizeX;
	}

	public NeuralNetWorkDL4J getNn() {
		return nn;
	}

	public void setNn(NeuralNetWorkDL4J nn) {
		this.nn = nn;
	}

	public double getBaseEpsilon() {
		return baseEpsilon;
	}

	public void setBaseEpsilon(double baseEpsilon) {
		this.baseEpsilon = baseEpsilon;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public int getnEpochs() {
		return nEpochs;
	}

	public void setnEpochs(int nEpochs) {
		this.nEpochs = nEpochs;
	}

	public void setSizeX(int sizeX) {
		this.sizeX = sizeX;
	}

	public int getSizeY() {
		return sizeY;
	}

	public void setSizeY(int sizeY) {
		this.sizeY = sizeY;
	}

	public int getNbAction() {
		return nbAction;
	}

}
