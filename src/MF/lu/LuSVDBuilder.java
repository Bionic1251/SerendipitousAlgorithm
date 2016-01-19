package MF.lu;

import annotation.Alpha;
import annotation.Threshold;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;

public class LuSVDBuilder extends LuSVDModelBuilder {
	@Inject
	public LuSVDBuilder(@Transient @Nonnull PreferenceSnapshot snapshot, @FeatureCount int featureCount, @InitialFeatureValue double initVal, @Threshold double threshold, @Nullable PreferenceDomain dom, @Alpha double alpha, @LearningRate double lrate, @RegularizationTerm double reg, StoppingCondition stop) {
		super(snapshot, featureCount, initVal, threshold, dom, alpha, lrate, reg, stop);
	}

	@Override
	protected double getDerivative(double a, double pop, double diff) {
		if (diff < 0) {
			return 0;
		}
		return pop * a / Math.abs(a);
	}
}
