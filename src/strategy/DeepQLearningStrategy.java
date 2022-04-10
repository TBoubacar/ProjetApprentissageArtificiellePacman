package strategy;

import java.util.ArrayList;

import agent.AgentAction;

import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;

import java.util.Random;


public class DeepQLearningStrategy extends NNQLearningStrats {

	private final int nbAction = 4;
	private NeuralNetWorkDL4J nn;
	private double baseEpsilon;
	private int nEpochs;
	private int batchSize;
	private int sizeX;
	private int sizeY;

	public DeepQLearningStrategy(double epsilon, double gamma, double alpha, int range, int nEpochs, int batchSize) {
		// TODO Auto-generated constructor stub
		super(epsilon, gamma, alpha);
		this.nn = new NeuralNetWorkDL4J(alpha, 0, this.nbAction, 2 );
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
			
			double[] encodedState = new double[this.nbAction];

			encodedState[genereEtatOfTable(state)] = 1;
			double[] outPut = this.nn.predict(encodedState);
			
			for(int a=0; a < this.nbAction; a++) {
								
				if(outPut[a%2] > maxQvalue) {
					maxQvalue = outPut[a%2];
					actionChoose = new AgentAction(a);
				} else if(outPut[a%2]  == maxQvalue) {
					if(random.nextBoolean()) {
						maxQvalue = outPut[a%2];
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
			double[] encodedState = new double[this.nbAction];
			encodedState[this.genereEtatOfTable(nextState)] = 1;
			double[] qValues_nextState = this.nn.predict(encodedState);

			if(qValues_nextState[0] > qValues_nextState[1]) {
				maxQvalue_nextState = qValues_nextState[0];
			} else {
				maxQvalue_nextState = qValues_nextState[1];
			}
		}
		
		double[] encodedState = new double[this.nbAction];
		encodedState[this.genereEtatOfTable(state)] = 1;
		
		double[] targetQ = this.nn.predict(encodedState);
		targetQ[action.get_direction()%2] = reward + gamma*maxQvalue_nextState;
		
		TrainExample trainExample = new TrainExample(encodedState, targetQ);
		this.trainExamples.add(trainExample);
	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// TODO Auto-generated method stub
		this.nn.fit(trainExamples, this.nEpochs, this.batchSize, this.baseEpsilon);
	}
	
	public int genereEtatOfTable(PacmanGame state) {
		int etat = 0;
		for (int j = 1; j <= this.sizeY; j++) {
			for (int i = 1; i <= this.sizeX; i++) {
				if (state.pacman.get_position().getX() == i && state.pacman.get_position().getY() == j) {
					etat += 1;
				}
				else if (state.isGhostsAt(i,j)) {
					etat += 2;
				}
				else if (state.getMaze().isFood(i, j)) {
					etat += 3;
				}
				else if (state.getMaze().isCapsule(i, j)) {
					etat += 4;
				} else {
					etat += 0;
				}
			}
		}
		return etat % this.nbAction;
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
