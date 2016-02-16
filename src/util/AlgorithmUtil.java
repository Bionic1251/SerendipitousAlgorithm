package util;

import adamopoulos.AdaItemScorer;
import alg.pop.AlgPopItemScorer;
import annotation.*;
import funkSVD.lu.LuFunkSVDItemScorer;
import funkSVD.lu.LuUpdateRule;
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
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import pop.PopItemScorer;
import random.RandomItemScorer;

import java.util.HashMap;
import java.util.Map;

public class AlgorithmUtil {
	//these constants are overriden in the property file
	public static double THRESHOLD = 3.0;
	public static double ALPHA = 0.5;
	public static int ITERATION_COUNT = 200;
	public static int FEATURE_COUNT = 20;
	public static double LEARNING_RATE = 0.00001;
	public static double REGULARIZATION_TERM = 0.0001;

	public static double ZHENG_LEARNING_RATE = 0.0001;
	public static double ZHENG_REGULARIZATION_TERM = 0.001;

	public static double LU_LEARNING_RATE = 0.000001;
	public static double LU_REGULARIZATION_TERM = 0.00001;

	public static Map<Long, SparseVector> itemContentMap;

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

	public static LenskitConfiguration getAdaFunkSVD() {
		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(FunkSVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		adaSVD.set(LearningRate.class).to(LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(ITERATION_COUNT);
		adaSVD.set(Threshold.class).to(THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		adaSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return adaSVD;
	}

	public static LenskitConfiguration getAdaSVD() {
		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		adaSVD.set(LearningRate.class).to(LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(ITERATION_COUNT);
		adaSVD.set(Threshold.class).to(THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		adaSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return adaSVD;
	}

	public static LenskitConfiguration getZhengFunkSVD() {
		LenskitConfiguration zhengFunkSVD = new LenskitConfiguration();
		zhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		zhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengFunkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		zhengFunkSVD.set(LearningRate.class).to(LEARNING_RATE);
		zhengFunkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		zhengFunkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		zhengFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengFunkSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengFunkSVD;
	}

	private static LenskitConfiguration getZhengSVDTemplate() {
		LenskitConfiguration zhengSVD = new LenskitConfiguration();
		zhengSVD.bind(ItemScorer.class).to(ZhengSVDItemScorer.class);
		zhengSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		zhengSVD.set(LearningRate.class).to(ZHENG_LEARNING_RATE);
		zhengSVD.set(RegularizationTerm.class).to(ZHENG_REGULARIZATION_TERM);
		zhengSVD.set(IterationCount.class).to(ITERATION_COUNT);
		zhengSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengSVD;
	}

	public static LenskitConfiguration getZhengSVD() {
		LenskitConfiguration zhengSVD = getZhengSVDTemplate();
		zhengSVD.set(Alpha.class).to(1);
		zhengSVD.set(UnobservedItemsIncluded.class).to(true);
		zhengSVD.set(ContentBased.class).to(false);
		return zhengSVD;
	}

	public static LenskitConfiguration getZhengSVDContent() {
		LenskitConfiguration zhengSVD = getZhengSVDTemplate();
		zhengSVD.set(Alpha.class).to(1);
		zhengSVD.set(UnobservedItemsIncluded.class).to(true);
		zhengSVD.set(ContentBased.class).to(true);
		return zhengSVD;
	}

	public static LenskitConfiguration getZhengSVDObserved() {
		LenskitConfiguration zhengSVD = getZhengSVDTemplate();
		zhengSVD.set(Alpha.class).to(1);
		zhengSVD.set(UnobservedItemsIncluded.class).to(false);
		return zhengSVD;
	}

	public static LenskitConfiguration getZhengSVDBasic() {
		LenskitConfiguration zhengSVD  = getZhengSVDTemplate();
		zhengSVD.set(Alpha.class).to(0);
		return zhengSVD;
	}

	public static LenskitConfiguration getSVD() {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(FEATURE_COUNT);
		svd.set(LearningRate.class).to(LEARNING_RATE);
		svd.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		svd.set(IterationCount.class).to(ITERATION_COUNT);
		return svd;
	}

	private static LenskitConfiguration getLuSVD() {
		LenskitConfiguration luSVD = new LenskitConfiguration();
		luSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		luSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		luSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		luSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		luSVD.set(IterationCount.class).to(ITERATION_COUNT);
		luSVD.set(Threshold.class).to(THRESHOLD);
		luSVD.set(Alpha.class).to(ALPHA);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDBaysian() {
		LenskitConfiguration luFunkSVD = getLuSVD();
		luFunkSVD.set(LearningRate.class).to(LU_LEARNING_RATE);
		luFunkSVD.set(RegularizationTerm.class).to(LU_REGULARIZATION_TERM);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleBaysian.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getLuSVDHinge10000() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(LU_LEARNING_RATE);
		luSVD.set(RegularizationTerm.class).to(LU_REGULARIZATION_TERM);
		luSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		luSVD.set(NormMult.class).to(10000.0);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDHinge() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(LU_LEARNING_RATE * 1000);
		luSVD.set(RegularizationTerm.class).to(LU_REGULARIZATION_TERM * 1000);
		luSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		luSVD.set(NormMult.class).to(1.0);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDBasic() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(LU_LEARNING_RATE);
		luSVD.set(RegularizationTerm.class).to(LU_REGULARIZATION_TERM);
		luSVD.set(UpdateRule.class).to(LuUpdateRule.class);
		return luSVD;
	}

	private static LenskitConfiguration getLuFunkSVD() {
		LenskitConfiguration funkSVD = new LenskitConfiguration();
		funkSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorer.class);
		funkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		funkSVD.set(LearningRate.class).to(LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		funkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		funkSVD.set(Threshold.class).to(THRESHOLD);
		funkSVD.set(Alpha.class).to(ALPHA);
		return funkSVD;
	}

	public static LenskitConfiguration getLuFunkSVDBaysian() {
		LenskitConfiguration luFunkSVD = getLuFunkSVD();
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleBaysian.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getLuFunkSVDHinge() {
		LenskitConfiguration luFunkSVD = getLuFunkSVD();
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getFunkSVD() {
		LenskitConfiguration funkSVD = new LenskitConfiguration();
		funkSVD.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		funkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		funkSVD.set(LearningRate.class).to(LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		funkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		return funkSVD;
	}

	public static LenskitConfiguration getPop() {
		LenskitConfiguration pop = new LenskitConfiguration();
		pop.bind(ItemScorer.class).to(PopItemScorer.class);
		pop.set(Reverse.class).to(false);
		return pop;
	}

	public static LenskitConfiguration getReversePop() {
		LenskitConfiguration pop = new LenskitConfiguration();
		pop.bind(ItemScorer.class).to(PopItemScorer.class);
		pop.set(Reverse.class).to(true);
		return pop;
	}

	private static LenskitConfiguration getAlg() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(AlgPopItemScorer.class);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		alg.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		alg.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		alg.set(FeatureCount.class).to(FEATURE_COUNT);
		alg.set(LearningRate.class).to(LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(ITERATION_COUNT);
		return alg;
	}

	public static LenskitConfiguration getAlgAll() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(1.0);
		alg.set(DissimilarityWeight.class).to(1.0);
		alg.set(RelevanceWeight.class).to(2.0);
		return alg;
	}

	public static LenskitConfiguration getAlgDissimUnpop() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.3);
		alg.set(DissimilarityWeight.class).to(0.3);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getAlgDissimRel() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.0);
		alg.set(DissimilarityWeight.class).to(0.5);
		alg.set(RelevanceWeight.class).to(0.5);
		return alg;
	}

	public static LenskitConfiguration getAlgUnpopRel() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.6);
		alg.set(DissimilarityWeight.class).to(0.0);
		alg.set(RelevanceWeight.class).to(0.6);
		return alg;
	}

	public static LenskitConfiguration getAlgUnpop() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.8);
		alg.set(DissimilarityWeight.class).to(0.0);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getAlgDissim() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.0);
		alg.set(DissimilarityWeight.class).to(0.7);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getAlgSim() {
		LenskitConfiguration alg = getAlg();
		alg.set(UnpopWeight.class).to(0.0);
		alg.set(DissimilarityWeight.class).to(-0.1);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}
	// POP POPReverse
	public static Map<String, LenskitConfiguration> getMap() {
		Map<String, LenskitConfiguration> configurationMap = new HashMap<String, LenskitConfiguration>();
		configurationMap.put("POP", AlgorithmUtil.getPop());
		configurationMap.put("POPReverse", AlgorithmUtil.getReversePop());
		configurationMap.put("LuFunkSVDBaysian", AlgorithmUtil.getLuFunkSVDBaysian());
		configurationMap.put("LuFunkSVDHinge", AlgorithmUtil.getLuFunkSVDHinge());
		configurationMap.put("FunkSVD", AlgorithmUtil.getFunkSVD());
		configurationMap.put("LuSVDHinge", AlgorithmUtil.getLuSVDHinge());
		configurationMap.put("LuSVDHinge10000", AlgorithmUtil.getLuSVDHinge10000());
		configurationMap.put("LuSVDBasic", AlgorithmUtil.getLuSVDBasic());
		configurationMap.put("LuSVDBaysian", AlgorithmUtil.getLuSVDBaysian());
		configurationMap.put("SVD", AlgorithmUtil.getSVD());
		configurationMap.put("ZhengSVD", AlgorithmUtil.getZhengSVD());
		configurationMap.put("ZhengSVDContent", AlgorithmUtil.getZhengSVDContent());
		configurationMap.put("ZhengSVDObserved", AlgorithmUtil.getZhengSVDObserved());
		configurationMap.put("ZhengSVDBasic", AlgorithmUtil.getZhengSVDBasic());
		configurationMap.put("ZhengFunkSVD", AlgorithmUtil.getZhengFunkSVD());
		configurationMap.put("AdaSVD", AlgorithmUtil.getAdaSVD());
		configurationMap.put("AdaFunkSVD", AlgorithmUtil.getAdaFunkSVD());
		configurationMap.put("ItemItem", AlgorithmUtil.getItemItem());
		configurationMap.put("Random", AlgorithmUtil.getRandom());
		configurationMap.put("AlgAll", AlgorithmUtil.getAlgAll());
		configurationMap.put("AlgDissimUnpop", AlgorithmUtil.getAlgDissimUnpop());
		configurationMap.put("AlgDissimRel", AlgorithmUtil.getAlgDissimRel());
		configurationMap.put("AlgUnpopRel", AlgorithmUtil.getAlgUnpopRel());
		configurationMap.put("AlgUnpop", AlgorithmUtil.getAlgUnpop());
		configurationMap.put("AlgDissim", AlgorithmUtil.getAlgDissim());
		configurationMap.put("AlgSim", AlgorithmUtil.getAlgSim());
		return configurationMap;
	}
}
