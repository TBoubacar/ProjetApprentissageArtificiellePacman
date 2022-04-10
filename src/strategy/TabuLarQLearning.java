package strategy;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Map;
import java.util.Random;

import agent.AgentAction;

import motor.PacmanGame;

public class TabuLarQLearning  extends QLearningStrategy{
	private Hashtable<String, double[] > tabQLearning;
	private final int nbAction = 4;
	private int sizeX;
	private int sizeY;

	public TabuLarQLearning(double epsilon, double gamma, double alpha, int sizeX, int sizeY) {
		// TODO Auto-generated constructor stub
		super(epsilon, gamma, alpha);
		this.sizeX = sizeX;
		this.sizeY = sizeY;
		this.tabQLearning = new Hashtable<String, double[]>();
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		// TODO Auto-generated method stub
		AgentAction actionChoose; 
		String etat = this.putInTabQLearning(state);	//INITIALISATION DE LA TABLE
		
		if(Math.random() < epsilon) {
			do {
				int n = (int) (Math.random() * (this.nbAction));
				actionChoose = new AgentAction(n);				
			} while(! state.isLegalMove(state.pacman, actionChoose));
			
		} else {
			AgentAction actionAleatoire = this.chooseAleatoireAction(state, 0);
			
			if (actionAleatoire != null) {
				return actionAleatoire;
			} else {
				int indiceMax = 0;
				double max = this.tabQLearning.get(etat)[indiceMax];
				AgentAction actionAleatoire2;
				
				for (Map.Entry<String, double[]> val : this.tabQLearning.entrySet()) {
					if (val.getKey().equals(etat)) {
						for (int i = 1; i < val.getValue().length; i++) {
							
							if(max < val.getValue()[i]) {
								indiceMax = i;
								max = val.getValue()[i];
							}
							
							actionChoose = new AgentAction(indiceMax);
							actionAleatoire2 = this.chooseAleatoireAction(state, i+1);
							if (actionAleatoire2 != null && max == val.getValue()[i+1]) {
								if (new Random().nextBoolean()) {
									return actionAleatoire;
								} else {
									return actionAleatoire2;
								}
							}
						}
					}
				}
				
				actionChoose = new AgentAction(indiceMax);
			}
		}
		
		return actionChoose;
	}

	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		// TODO Auto-generated method stub 
		String etat = this.putInTabQLearning(state);		//MISE A JOUR DE LA TABLE
		String newEtat = this.putInTabQLearning(nextState);	//INITIALISATION DE LA TABLE
		double maxQnextState = this.tabQLearning.get(newEtat)[0];

		for (int i = 0; i < this.tabQLearning.get(newEtat).length; i++) {
			if (maxQnextState < this.tabQLearning.get(newEtat)[i]) {
				maxQnextState = this.tabQLearning.get(newEtat)[i];
			}
		}
		
		this.tabQLearning.get(etat)[action.get_direction()] = (1-alpha)* this.tabQLearning.get(etat)[action.get_direction()] + alpha *( reward + gamma * maxQnextState);
		
		for (Map.Entry<String, double[]> val : this.tabQLearning.entrySet()) {
			System.out.print("TabQLearning [NORTH : 0 | SOUTH = 1 | EAST = 2 | WEST = 3 ] => (" + val.getKey() + ") : [");
			for (int i = 0; i < val.getValue().length; i++) {
				System.out.print(val.getValue()[i] + "|");
			}
			System.out.println("]");
		}
	}
	
	/*---		AJOUT DE CODE PLUS BAS		---*/

	public ArrayList<Integer>  isLegalMoveTabId(PacmanGame state, int idActionBegin) {
		// TODO Auto-generated method stub
		ArrayList<Integer> tabIdLegalMove = new ArrayList<Integer>();
		String etat = this.putInTabQLearning(state);
		AgentAction action;
		for (int i = idActionBegin; i < this.tabQLearning.get(etat).length; i++) {	// ON REGARDE LA TABLE CONTENANT L'ENSEMBLE DES ACTIONS POSSIBLES POUR CET ETAT, SI L'ACTION EST POSSIBLE ON LE MET DANS LE TABLEAU SINON ON PASSE
			action = new AgentAction(i);
			if (state.isLegalMove(state.pacman, action)) {
				tabIdLegalMove.add(i);
			}
		}
		if (tabIdLegalMove.size() == 0) return null;
		return tabIdLegalMove;
	}	//	CETTE METHODE ME PERMET DE RECUPERER L'ID DES ACTIONS POSSIBLES DANS UN TABLEAU. S'IL N'Y A PAS D'ACTION POSSIBLE A FAIRE, ON RETURN NULL
	
	public AgentAction chooseAleatoireAction(PacmanGame state, int idActionBegin) {
		// TODO Auto-generated method stub
		ArrayList<Integer> tabIdLegalMove = this.isLegalMoveTabId(state, idActionBegin);

		if (tabIdLegalMove != null) {
			String etat = this.putInTabQLearning(state);
			boolean isSameAction = true;
			int firstIdOfLegalAction = tabIdLegalMove.get(0);
			int idOfLegalAction;
			int numeroAction;
			
			double val = this.tabQLearning.get(etat)[firstIdOfLegalAction];
			for (int i = 1; i < tabIdLegalMove.size(); i++) {
				idOfLegalAction = tabIdLegalMove.get(i); 
				if (this.tabQLearning.get(etat)[idOfLegalAction] != val) {
					isSameAction = false;
					break;
				}
			}
			
			if (isSameAction == true) {
				int indiceAction = (int) (Math.random() * (tabIdLegalMove.size()));
				numeroAction = tabIdLegalMove.get(indiceAction);
				return new AgentAction(numeroAction);
			}
		}
		return null;
	}	// CETTE METHODE ME PERMET DE RETOURNER UNE ACTION ALEATOIRE PARMI LES VALEURS POSSIBLES. DANS LE CAS OU IL N'Y A PAS DE CHOIX ALEATOIRE A FAIRE, ON RETURN NULL
	
	public String genereEtatOfTable(PacmanGame state) {
		String etat = "";
		for (int j = 1; j <= this.sizeY; j++) {
			for (int i = 1; i <= this.sizeX; i++) {
				if (state.pacman.get_position().getX() == i && state.pacman.get_position().getY() == j) {
					etat += "1";
				}
				else if (state.isGhostsAt(i,j)) {
					etat += "2";
				}
				else if (state.getMaze().isFood(i, j)) {
					etat += "3";
				}
				else if (state.getMaze().isCapsule(i, j)) {
					etat += "4";
				} else {
					etat += "0";
				}
			}
		}
		return etat;
	}
	
	public String putInTabQLearning(PacmanGame state) {
		double[] tabAction = new double[this.nbAction];
		for (int i = 0; i < this.nbAction; i++) {
			tabAction[i] = 0;
		}
		String etat = this.genereEtatOfTable(state);
		if (! this.tabQLearning.containsKey(etat)) {
			this.tabQLearning.put(etat, tabAction);			
		}
		return etat;
	}
	
	public int getSizeX() {
		return sizeX;
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

	public Hashtable<String, double[]> getTabQLearning() {
		return tabQLearning;
	}

	public void setTabQLearning(Hashtable<String, double[]> tabQLearning) {
		this.tabQLearning = tabQLearning;
	}

	public int getNbAction() {
		return nbAction;
	}
}
