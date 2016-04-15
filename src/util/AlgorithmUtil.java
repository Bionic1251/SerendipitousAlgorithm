package util;

import adamopoulos.AdaItemScorer;
import annotation.*;
import content.ContentItemScorer;
import content.SerContentItemScorer;
import funkSVD.lu.LuFunkSVDItemScorer;
import funkSVD.lu.LuUpdateRule;
import funkSVD.lu.LuUpdateRuleBaysian;
import funkSVD.lu.LuUpdateRuleHinge;
import funkSVD.zheng.ZhengFunkSVDItemScorer;
import lc.advanced.LCAdvancedItemScorer;
import lc.basic.LCITest;
import lc.basic.LCItemScorer;
import lc.investigation.NonPersInvestigationItemScorer;
import lc.investigation.PersInvestigationItemScorer;
import mf.baseline.SVDItemScorer;
import mf.lu.LuSVDItemScorer;
import mf.pureSVD.PureSVDItemScorer;
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
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import pop.PopItemScorer;
import random.CompletelyRandomItemScorer;
import random.RandomItemScorer;

import java.util.HashMap;
import java.util.Map;

public class AlgorithmUtil {

	public static LenskitConfiguration getRandom() {
		LenskitConfiguration rnd = new LenskitConfiguration();
		rnd.bind(ItemScorer.class).to(RandomItemScorer.class);
		return rnd;
	}

	public static LenskitConfiguration getCompletelyRandom() {
		LenskitConfiguration rnd = new LenskitConfiguration();
		rnd.bind(ItemScorer.class).to(CompletelyRandomItemScorer.class);
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
		adaSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		adaSVD.set(LearningRate.class).to(Settings.LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		adaSVD.set(R_Threshold.class).to(Settings.R_THRESHOLD);
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
		adaSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		adaSVD.set(LearningRate.class).to(Settings.LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		adaSVD.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		adaSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return adaSVD;
	}

	public static LenskitConfiguration getAdaPureSVD() {
		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(PureSVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		adaSVD.set(LearningRate.class).to(Settings.ZHENG_LEARNING_RATE);
		adaSVD.set(RegularizationTerm.class).to(Settings.ZHENG_REGULARIZATION_TERM);
		adaSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_PURE_SVD);
		adaSVD.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		adaSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return adaSVD;
	}

	public static LenskitConfiguration getZhengFunkSVD() {
		LenskitConfiguration zhengFunkSVD = new LenskitConfiguration();
		zhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		zhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengFunkSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		zhengFunkSVD.set(LearningRate.class).to(Settings.LEARNING_RATE);
		zhengFunkSVD.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		zhengFunkSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_PURE_SVD);
		zhengFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengFunkSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengFunkSVD;
	}

	private static LenskitConfiguration getZhengSVDTemplate() {
		LenskitConfiguration zhengSVD = new LenskitConfiguration();
		zhengSVD.bind(ItemScorer.class).to(ZhengSVDItemScorer.class);
		zhengSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		zhengSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		zhengSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		zhengSVD.set(LearningRate.class).to(Settings.ZHENG_LEARNING_RATE);
		zhengSVD.set(RegularizationTerm.class).to(Settings.ZHENG_REGULARIZATION_TERM);
		zhengSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_PURE_SVD);
		zhengSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		zhengSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return zhengSVD;
	}

	public static LenskitConfiguration getZhengSVDContent() {
		LenskitConfiguration zhengSVD = getZhengSVDTemplate();
		return zhengSVD;
	}

	public static LenskitConfiguration getSVD() {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		svd.set(LearningRate.class).to(Settings.LEARNING_RATE);
		svd.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		svd.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return svd;
	}

	public static LenskitConfiguration getSVDManyFeatures() {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(200);
		svd.set(LearningRate.class).to(Settings.LEARNING_RATE);
		svd.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		svd.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return svd;
	}

	public static LenskitConfiguration getSVDManyIterations() {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		svd.set(LearningRate.class).to(Settings.LEARNING_RATE);
		svd.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		svd.set(IterationCount.class).to(2000);
		return svd;
	}

	public static LenskitConfiguration getSVDLearningRate() {
		LenskitConfiguration svd = new LenskitConfiguration();
		svd.bind(ItemScorer.class).to(SVDItemScorer.class);
		svd.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		svd.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		svd.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		svd.set(LearningRate.class).to(0.0001);
		svd.set(RegularizationTerm.class).to(0.001);
		svd.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return svd;
	}

	private static LenskitConfiguration getLuSVD() {
		LenskitConfiguration luSVD = new LenskitConfiguration();
		luSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		luSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		luSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		luSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		luSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_SPR);
		luSVD.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		luSVD.set(Alpha.class).to(Settings.ALPHA);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDBaysian() {
		LenskitConfiguration luFunkSVD = getLuSVD();
		luFunkSVD.set(LearningRate.class).to(Settings.LU_LEARNING_RATE);
		luFunkSVD.set(RegularizationTerm.class).to(Settings.LU_REGULARIZATION_TERM);
		luFunkSVD.set(UpdateRule.class).to(LuUpdateRuleBaysian.class);
		return luFunkSVD;
	}

	public static LenskitConfiguration getLuSVDHinge10000() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(Settings.LU_LEARNING_RATE);
		luSVD.set(RegularizationTerm.class).to(Settings.LU_REGULARIZATION_TERM);
		luSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		luSVD.set(NormMult.class).to(10000.0);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDHinge() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(Settings.LU_LEARNING_RATE * 1000);
		luSVD.set(RegularizationTerm.class).to(Settings.LU_REGULARIZATION_TERM * 1000);
		luSVD.set(UpdateRule.class).to(LuUpdateRuleHinge.class);
		luSVD.set(NormMult.class).to(1.0);
		return luSVD;
	}

	public static LenskitConfiguration getLuSVDBasic() {
		LenskitConfiguration luSVD = getLuSVD();
		luSVD.set(LearningRate.class).to(Settings.LU_LEARNING_RATE);
		luSVD.set(RegularizationTerm.class).to(Settings.LU_REGULARIZATION_TERM);
		luSVD.set(UpdateRule.class).to(LuUpdateRule.class);
		return luSVD;
	}

	private static LenskitConfiguration getLuFunkSVD() {
		LenskitConfiguration funkSVD = new LenskitConfiguration();
		funkSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorer.class);
		funkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		funkSVD.set(LearningRate.class).to(Settings.LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		funkSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_SPR);
		funkSVD.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		funkSVD.set(Alpha.class).to(Settings.ALPHA);
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
		//funkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		funkSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		funkSVD.set(LearningRate.class).to(Settings.LEARNING_RATE);
		funkSVD.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		funkSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
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

	private static LenskitConfiguration getLC() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(LCItemScorer.class);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		alg.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		alg.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		alg.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		alg.set(LearningRate.class).to(Settings.LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return alg;
	}

	public static LenskitConfiguration getLCRDU() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(1.1);
		alg.set(DissimilarityWeight.class).to(1.1);
		alg.set(RelevanceWeight.class).to(1.1);
		return alg;
	}

	public static LenskitConfiguration getLCDU() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(0.31);
		alg.set(DissimilarityWeight.class).to(0.31);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getLCRD() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(0.0);
		alg.set(DissimilarityWeight.class).to(0.51);
		alg.set(RelevanceWeight.class).to(0.51);
		return alg;
	}

	public static LenskitConfiguration getLCRU() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(0.6);
		alg.set(DissimilarityWeight.class).to(0.0);
		alg.set(RelevanceWeight.class).to(0.6);
		return alg;
	}

	public static LenskitConfiguration getLCU() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(0.81);
		alg.set(DissimilarityWeight.class).to(0.0);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getLCD() {
		LenskitConfiguration alg = getLC();
		alg.set(UnpopWeight.class).to(0.0);
		alg.set(DissimilarityWeight.class).to(0.7);
		alg.set(RelevanceWeight.class).to(0.0);
		return alg;
	}

	public static LenskitConfiguration getNonPersInvestigation() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(NonPersInvestigationItemScorer.class);
		return alg;
	}

	public static LenskitConfiguration getPersInvestigation() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(PersInvestigationItemScorer.class);
		return alg;
	}

	public static LenskitConfiguration getLCSVD() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(LCAdvancedItemScorer.class);
		alg.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		alg.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		alg.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		alg.set(LearningRate.class).to(Settings.LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		alg.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		alg.set(D_Threshold.class).to(Settings.D_THRESHOLD);
		alg.set(U_Threshold.class).to(Settings.U_THRESHOLD);
		return alg;
	}

	public static LenskitConfiguration getLCPureSVD() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(LCAdvancedItemScorer.class);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(PureSVDItemScorer.class);
		alg.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		alg.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		alg.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		alg.set(LearningRate.class).to(Settings.ZHENG_LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(Settings.ZHENG_REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(Settings.ITERATION_COUNT_PURE_SVD);
		alg.set(R_Threshold.class).to(Settings.R_THRESHOLD);
		alg.set(D_Threshold.class).to(Settings.D_THRESHOLD);
		alg.set(U_Threshold.class).to(Settings.U_THRESHOLD);
		return alg;
	}

	public static LenskitConfiguration getPureSVD() {
		LenskitConfiguration pureSVD = new LenskitConfiguration();
		pureSVD.bind(ItemScorer.class).to(PureSVDItemScorer.class);
		pureSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		pureSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		pureSVD.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		pureSVD.set(LearningRate.class).to(Settings.ZHENG_LEARNING_RATE);
		pureSVD.set(RegularizationTerm.class).to(Settings.ZHENG_REGULARIZATION_TERM);
		pureSVD.set(IterationCount.class).to(Settings.ITERATION_COUNT_PURE_SVD);
		return pureSVD;
	}

	private static LenskitConfiguration getLCTest() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(LCITest.class);
		return alg;
	}

	public static LenskitConfiguration getContent() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(ContentItemScorer.class);
		alg.set(TFIDF.class).to(false);
		return alg;
	}

	public static LenskitConfiguration getTFIDF() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(ContentItemScorer.class);
		alg.set(TFIDF.class).to(true);
		return alg;
	}

	public static LenskitConfiguration getSerContent() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(SerContentItemScorer.class);
		alg.set(TFIDF.class).to(false);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		alg.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		alg.set(LearningRate.class).to(Settings.LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return alg;
	}

	public static LenskitConfiguration getSerTFIDF() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(SerContentItemScorer.class);
		alg.set(TFIDF.class).to(true);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		alg.set(FeatureCount.class).to(Settings.FEATURE_COUNT);
		alg.set(LearningRate.class).to(Settings.LEARNING_RATE);
		alg.set(RegularizationTerm.class).to(Settings.REGULARIZATION_TERM);
		alg.set(IterationCount.class).to(Settings.ITERATION_COUNT_SVD);
		return alg;
	}

	public static LenskitConfiguration getSerUB() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(SerContentItemScorer.class);
		alg.set(TFIDF.class).to(false);
		alg.bind(RatingPredictor.class, ItemScorer.class).to(UserUserItemScorer.class);
		alg.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		return alg;
	}

	public static LenskitConfiguration getSerPop() {
		LenskitConfiguration alg = new LenskitConfiguration();
		alg.bind(ItemScorer.class).to(PopItemScorer.class);
		alg.set(TFIDF.class).to(false);
		return alg;
	}

	public static Map<String, LenskitConfiguration> getMap() {
		Map<String, LenskitConfiguration> configurationMap = new HashMap<String, LenskitConfiguration>();
		//Funk
		configurationMap.put("LuFunkSVDBaysian", AlgorithmUtil.getLuFunkSVDBaysian());
		configurationMap.put("LuFunkSVDHinge", AlgorithmUtil.getLuFunkSVDHinge());
		configurationMap.put("FunkSVD", AlgorithmUtil.getFunkSVD());
		configurationMap.put("ZhengFunkSVD", AlgorithmUtil.getZhengFunkSVD());
		configurationMap.put("AdaFunkSVD", AlgorithmUtil.getAdaFunkSVD());

		configurationMap.put("POP", AlgorithmUtil.getPop());
		configurationMap.put("POPReverse", AlgorithmUtil.getReversePop());

		//Lu
		configurationMap.put("LuSVDHinge", AlgorithmUtil.getLuSVDHinge());
		configurationMap.put("LuSVDHinge10000", AlgorithmUtil.getLuSVDHinge10000());
		configurationMap.put("LuSVDBasic", AlgorithmUtil.getLuSVDBasic());
		configurationMap.put("LuSVDBaysian", AlgorithmUtil.getLuSVDBaysian());

		configurationMap.put("SVD", AlgorithmUtil.getSVD());
		configurationMap.put("getSVDLearningRate", AlgorithmUtil.getSVDLearningRate());
		configurationMap.put("getSVDManyFeatures", AlgorithmUtil.getSVDManyFeatures());
		configurationMap.put("getSVDManyIterations", AlgorithmUtil.getSVDManyIterations());

		configurationMap.put("ZhengSVD", AlgorithmUtil.getZhengSVDContent());
		configurationMap.put("AdaSVD", AlgorithmUtil.getAdaSVD());
		configurationMap.put("AdaPureSVD", AlgorithmUtil.getAdaPureSVD());
		configurationMap.put("ItemItem", AlgorithmUtil.getItemItem());
		configurationMap.put("Random", AlgorithmUtil.getRandom());
		configurationMap.put("CRandom", AlgorithmUtil.getCompletelyRandom());
		configurationMap.put("Content", AlgorithmUtil.getContent());
		configurationMap.put("tfidf", AlgorithmUtil.getTFIDF());
		configurationMap.put("SerContent", AlgorithmUtil.getSerContent());

		configurationMap.put("LCRDU", AlgorithmUtil.getLCRDU());
		configurationMap.put("LCDU", AlgorithmUtil.getLCDU());
		configurationMap.put("LCRD", AlgorithmUtil.getLCRD());
		configurationMap.put("LCRU", AlgorithmUtil.getLCRU());
		configurationMap.put("LCU", AlgorithmUtil.getLCU());
		configurationMap.put("LCD", AlgorithmUtil.getLCD());
		configurationMap.put("LCSVD", AlgorithmUtil.getLCSVD());
		configurationMap.put("LCTest", AlgorithmUtil.getLCTest());

		configurationMap.put("LCPureSVD", AlgorithmUtil.getLCPureSVD());
		configurationMap.put("PureSVD", AlgorithmUtil.getPureSVD());
		configurationMap.put("Investigation", AlgorithmUtil.getNonPersInvestigation());
		configurationMap.put("Investigation_per_user", AlgorithmUtil.getPersInvestigation());
		return configurationMap;
	}
}
