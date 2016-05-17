package ser.funkSer;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FunkSVDUpdater;
import org.grouplens.lenskit.mf.funksvd.TrainingEstimator;

import javax.annotation.Nullable;
import javax.inject.Inject;
import java.io.Serializable;

@Shareable
public class SerFunkSVDUpdateRule implements Serializable {
	private static final long serialVersionUID = 2L;

	private final double learningRate;
	private final double trainingRegularization;
	private final ItemScorer baseline;
	private final StoppingCondition stoppingCondition;
	@Nullable
	private final PreferenceDomain domain;

	/**
	 * Construct a new FunkSVD configuration.
	 *
	 * @param lrate The learning rate.
	 * @param reg   The regularization term.
	 * @param stop  The stopping condition
	 */
	@Inject
	public SerFunkSVDUpdateRule(@LearningRate double lrate,
							 @RegularizationTerm double reg,
							 @BaselineScorer ItemScorer bl,
							 @Nullable PreferenceDomain dom,
							 StoppingCondition stop) {
		learningRate = lrate;
		trainingRegularization = reg;
		baseline = bl;
		domain = dom;
		stoppingCondition = stop;
	}

	/**
	 * Create an estimator to use while training the recommender.
	 *
	 * @return The estimator to use.
	 */
	public SerTrainingEstimator makeEstimator(PreferenceSnapshot snapshot) {
		return new SerTrainingEstimator(snapshot, baseline, domain);
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double getTrainingRegularization() {
		return trainingRegularization;
	}

	public StoppingCondition getStoppingCondition() {
		return stoppingCondition;
	}

	@Nullable
	public PreferenceDomain getDomain() {
		return domain;
	}

	public TrainingLoopController getTrainingLoopController() {
		return stoppingCondition.newLoop();
	}

	public SerFunkSVDUpdater createUpdater() {
		return new SerFunkSVDUpdater(this);
	}
}
