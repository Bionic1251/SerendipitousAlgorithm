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
public class LuUpdateRuleBaysian extends LuFunkSVDUpdateRule {
	@Inject
	public LuUpdateRuleBaysian(@LearningRate double lrate, @RegularizationTerm double reg,
							   @BaselineScorer ItemScorer bl, @Nullable PreferenceDomain dom) {
		super(lrate, reg, bl, dom);
		System.out.println(LuUpdateRuleBaysian.class);
	}

	public double getFunctionVal(double diff, double pop) {
		return Math.log(1 + Math.exp(diff)) * pop;
	}

	protected double getDerivative(double a, double pop, double diff) {
		return a * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
	}
}
