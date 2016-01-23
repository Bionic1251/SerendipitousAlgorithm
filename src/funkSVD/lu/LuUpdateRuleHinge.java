package funkSVD.lu;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;

import javax.annotation.Nullable;
import javax.inject.Inject;

@Shareable
public class LuUpdateRuleHinge extends LuFunkSVDUpdateRule {
	@Inject
	public LuUpdateRuleHinge(@LearningRate double lrate, @RegularizationTerm double reg,
							 @BaselineScorer ItemScorer bl, @Nullable PreferenceDomain dom) {
		super(lrate, reg, bl, dom);
		System.out.println(LuUpdateRuleHinge.class);
	}

	public double getFunctionVal(double diff, double pop) {
		return Math.max(0, diff) * pop;
	}

	protected double getDerivative(double a, double pop, double diff) {
		if (diff < 0) {
			return 0;
		}
		return pop * a;
	}
}
