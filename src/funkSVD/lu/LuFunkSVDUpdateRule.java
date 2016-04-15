package funkSVD.lu;

import funkSVD.MyTrainingEstimator;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.mf.funksvd.TrainingEstimator;

import javax.annotation.Nullable;
import javax.inject.Inject;
import java.io.Serializable;


public abstract class LuFunkSVDUpdateRule implements Serializable {
	private final double learningRate;
	private final double trainingRegularization;
	private final ItemScorer baseline;
	private final int RANGE = 10000;

	private final PreferenceDomain domain;

	public LuFunkSVDUpdateRule(double lrate,
							   double reg,
							   ItemScorer bl,
							   PreferenceDomain dom) {
		learningRate = lrate;
		trainingRegularization = reg;
		baseline = bl;
		domain = dom;
	}

	public MyTrainingEstimator makeEstimator(PreferenceSnapshot snapshot) {
		return new MyTrainingEstimator(snapshot, baseline, domain);
	}

	private double getUpdate(double a, double pop, double diff, double value) {
		double derivativeVal = getDerivative(a, pop, diff);
		return (derivativeVal - trainingRegularization * value) * learningRate;
	}

	private boolean isInRange(double value) {
		return value < RANGE && value > -RANGE;
	}

	public double getNextStep(double a, double pop, double diff, double value) {
		double update = getUpdate(a, pop, diff, value);
		double newVal = value + update;
		if (Double.isNaN(newVal) || Double.isInfinite(newVal) || !isInRange(newVal)) {
			return 0.0;
		}
		return update;
	}

	public abstract double getFunctionVal(double diff, double pop);
	protected abstract double getDerivative(double a, double pop, double diff);
}
