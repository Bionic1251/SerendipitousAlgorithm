package lc.investigation;

public abstract class Optimizer {
	protected double r;
	protected double d;
	protected double u;

	protected final static double DEFAULT_VAL = 0.33;
	protected double wr = DEFAULT_VAL;
	protected double wd = DEFAULT_VAL;
	protected double wu = DEFAULT_VAL;

	protected final static int ITERATION_COUNT = 1000000;

	protected final static double LEARNING_RATE = 0.00001;
	protected final static double REG_TERM = 0.2;
	protected final static double ALPHA = 1.8;


	protected double func() {
		return wr * r + wd * d + wu * u;
	}

	public WeightTriple getWeights() {
		return new WeightTriple(wr, wd, wu);
	}

	public void optimize() {
		//System.out.println("Optimizing");
		for (int i = 0; i < ITERATION_COUNT; i++) {
			printFunction();
			updateWeights();
		}
	}

	private void printFunction() {
		//System.out.println("Function " + func());
		//System.out.println("Weights wr=" + wr + " wd=" + wd + " wu=" + wu + " sum=" + (wr + wd + wu));
	}

	protected abstract void updateWeights();
}
