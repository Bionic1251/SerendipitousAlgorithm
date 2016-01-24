package util;

import annotation.Alpha;
import annotation.Threshold;
import annotation.UpdateRule;
import funkSVD.lu.LuFunkSVDItemScorer;
import funkSVD.lu.LuUpdateRuleBaysian;
import funkSVD.lu.LuUpdateRuleHinge;
import funkSVD.zheng.ZhengFunkSVDItemScorer;
import mf.baseline.SVDItemScorer;
import mf.lu.LuSVDItemScorer;
import mf.zheng.ZhengSVDItemScorer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import pop.PopItemScorer;

public class AlgorithmUtil {
	private static final double THRESHOLD = 3.0;
	private static final double ALPHA = 0.5;
	private static final int ITERATION_COUNT = 20;

	public static LenskitConfiguration getZhengFunkSVD(int featureCount) {
		LenskitConfiguration zhengFunkSVD = new LenskitConfiguration();
		zhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		zhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengFunkSVD.set(FeatureCount.class).to(featureCount);
		zhengFunkSVD.set(LearningRate.class).to(0.00001);
		zhengFunkSVD.set(RegularizationTerm.class).to(0.0001);
		zhengFunkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		zhengFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengFunkSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengFunkSVD;
	}

	public static LenskitConfiguration getZhengSVD(int featureCount) {
		LenskitConfiguration zhengSVD = new LenskitConfiguration();
		zhengSVD.bind(ItemScorer.class).to(ZhengSVDItemScorer.class);
		zhengSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengSVD.set(FeatureCount.class).to(featureCount);
		zhengSVD.set(LearningRate.class).to(0.00001);
		zhengSVD.set(RegularizationTerm.class).to(0.0001);
		zhengSVD.set(IterationCount.class).to(ITERATION_COUNT);
		zhengSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengSVD;
	}

	public static LenskitConfiguration getSVD(int featureCount) {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(featureCount);
		svd.set(LearningRate.class).to(0.00001);
		svd.set(RegularizationTerm.class).to(0.0001);
		svd.set(IterationCount.class).to(ITERATION_COUNT);
		return svd;
	}

	private static LenskitConfiguration getLuSVD(int featureCount) {
		LenskitConfiguration luSVD = new LenskitConfiguration();
		luSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		luSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		luSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		luSVD.set(FeatureCount.class).to(featureCount);
		luSVD.set(LearningRate.class).to(0.00001);
		luSVD.set(RegularizationTerm.class).to(0.0001);
		luSVD.set(IterationCount.class).to(ITERATION_COUNT);
		luSVD.set(Threshold.class).to(THRESHOLD);
		luSVD.set(Alpha.class).to(ALPHA);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDBaysian(int featureCount) {
		LenskitConfiguration luFunkSVD = getLuSVD(featureCount);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleBaysian.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getLuSVDHinge(int featureCount) {
		LenskitConfiguration luFunkSVD = getLuSVD(featureCount);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		return luFunkSVD;
	}

	private static LenskitConfiguration getLuFunkSVD(int featureCount) {
		LenskitConfiguration funkSVD = new LenskitConfiguration();
		funkSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorer.class);
		funkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(featureCount);
		funkSVD.set(LearningRate.class).to(0.00001);
		funkSVD.set(RegularizationTerm.class).to(0.0001);
		funkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		funkSVD.set(Threshold.class).to(THRESHOLD);
		funkSVD.set(Alpha.class).to(ALPHA);
		return funkSVD;
	}

	public static LenskitConfiguration getLuFunkSVDBaysian(int featureCount) {
		LenskitConfiguration luFunkSVD = getLuFunkSVD(featureCount);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleBaysian.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getLuFunkSVDHinge(int featureCount) {
		LenskitConfiguration luFunkSVD = getLuFunkSVD(featureCount);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getFunkSVD(int featureCount) {
		LenskitConfiguration funkSVD = new LenskitConfiguration();
		funkSVD.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		funkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(featureCount);
		funkSVD.set(LearningRate.class).to(0.00001);
		funkSVD.set(RegularizationTerm.class).to(0.0001);
		funkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		return funkSVD;
	}

	public static LenskitConfiguration getPop() {
		LenskitConfiguration pop = new LenskitConfiguration();
		pop.bind(ItemScorer.class).to(PopItemScorer.class);
		return pop;
	}
}
