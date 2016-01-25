package util;

import adamopoulos.AdaItemScorer;
import annotation.Alpha;
import annotation.RatingPredictor;
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
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import pop.PopItemScorer;
import random.RandomItemScorer;

import java.util.HashMap;
import java.util.Map;

public class AlgorithmUtil {
	private static final double THRESHOLD = 3.0;
	private static final double ALPHA = 0.5;
	private static final int ITERATION_COUNT = 100;
	private static double LEARNING_RATE = 0.0001;
	private static double REGULARIZATION_TERM = 0.001;

	public static LenskitConfiguration getRandom() {
		LenskitConfiguration rnd = new LenskitConfiguration();
		rnd.bind(ItemScorer.class).to(RandomItemScorer.class);
		return rnd;
	}

	public static LenskitConfiguration getItemItem() {
		LenskitConfiguration itemItem = new LenskitConfiguration();
		itemItem.bind(ItemScorer.class).to(ItemItemScorer.class);
		itemItem.bind(BaselineScorer.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		itemItem.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return itemItem;
	}

	public static LenskitConfiguration getAdaFunkSVD(int featureCount) {
		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(FunkSVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(featureCount);
		adaSVD.set(LearningRate.class).to(LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(ITERATION_COUNT);
		adaSVD.set(Threshold.class).to(THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		return adaSVD;
	}

	public static LenskitConfiguration getAdaSVD(int featureCount) {
		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(featureCount);
		adaSVD.set(LearningRate.class).to(LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(ITERATION_COUNT);
		adaSVD.set(Threshold.class).to(THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		return adaSVD;
	}

	public static LenskitConfiguration getZhengFunkSVD(int featureCount) {
		LenskitConfiguration zhengFunkSVD = new LenskitConfiguration();
		zhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		zhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengFunkSVD.set(FeatureCount.class).to(featureCount);
		zhengFunkSVD.set(LearningRate.class).to(LEARNING_RATE);
		zhengFunkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
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
		zhengSVD.set(LearningRate.class).to(LEARNING_RATE);
		zhengSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
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
		svd.set(LearningRate.class).to(LEARNING_RATE);
		svd.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		svd.set(IterationCount.class).to(ITERATION_COUNT);
		return svd;
	}

	private static LenskitConfiguration getLuSVD(int featureCount) {
		LenskitConfiguration luSVD = new LenskitConfiguration();
		luSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		luSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		luSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		luSVD.set(FeatureCount.class).to(featureCount);
		luSVD.set(LearningRate.class).to(LEARNING_RATE);
		luSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
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
		funkSVD.set(LearningRate.class).to(LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
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
		funkSVD.set(LearningRate.class).to(LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		funkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		return funkSVD;
	}

	public static LenskitConfiguration getPop() {
		LenskitConfiguration pop = new LenskitConfiguration();
		pop.bind(ItemScorer.class).to(PopItemScorer.class);
		return pop;
	}

	public static Map<String, LenskitConfiguration> getMap(int featureCount){
		Map<String, LenskitConfiguration> configurationMap = new HashMap<String, LenskitConfiguration>();
		configurationMap.put("POP", AlgorithmUtil.getPop());
		configurationMap.put("LuFunkSVDBaysian", AlgorithmUtil.getLuFunkSVDBaysian(featureCount));
		configurationMap.put("LuFunkSVDHinge", AlgorithmUtil.getLuFunkSVDHinge(featureCount));
		configurationMap.put("FunkSVD", AlgorithmUtil.getFunkSVD(featureCount));
		configurationMap.put("LuSVDHinge", AlgorithmUtil.getLuSVDHinge(featureCount));
		configurationMap.put("LuSVDBaysian", AlgorithmUtil.getLuSVDBaysian(featureCount));
		configurationMap.put("SVD", AlgorithmUtil.getSVD(featureCount));
		configurationMap.put("ZhengSVD", AlgorithmUtil.getZhengSVD(featureCount));
		configurationMap.put("ZhengFunkSVD", AlgorithmUtil.getZhengFunkSVD(featureCount));
		configurationMap.put("AdaSVD", AlgorithmUtil.getAdaSVD(featureCount));
		configurationMap.put("AdaFunkSVD", AlgorithmUtil.getAdaFunkSVD(featureCount));
		configurationMap.put("ItemItem", AlgorithmUtil.getItemItem());
		configurationMap.put("Random", AlgorithmUtil.getRandom());
		return configurationMap;
	}
}
