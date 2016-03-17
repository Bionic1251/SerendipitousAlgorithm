package lc.investigation_per_user;

import annotation.D_Threshold;
import annotation.R_Threshold;
import annotation.U_Threshold;
import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.AverageAggregate;
import util.ContentAverageDissimilarity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.io.File;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class InvestigationPerUserModelBuilder implements Provider<InvestigationPerUserModel> {
	private final PopModel popModel;
	private final PreferenceSnapshot snapshot;
	private final double rThreshold;
	private final double dThreshold;
	private final double uThreshold;
	private Map<Long, SparseVector> userItemDissimilarityMap;

	private final double learningRate = 0.0001;
	private final double regularizationTerm = 0.001;
	private double pCoeff = 0.001;
	private final int iterationCount = 10;
	private double defaultVal = 0.33;

	private final Map<Long, WeightAggregate> weightMap = new HashMap<Long, WeightAggregate>();
	private final Map<Long, AverageAggregate> userThresholdMap;

	@Inject
	public InvestigationPerUserModelBuilder(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot,
											@R_Threshold double rThreshold, @D_Threshold double dThreshold, @U_Threshold double uThreshold) {
		this.popModel = popModel;
		this.snapshot = snapshot;
		this.rThreshold = rThreshold;
		this.dThreshold = dThreshold;
		this.uThreshold = uThreshold;
		ContentAverageDissimilarity contentAverageDissimilarity = ContentAverageDissimilarity.getInstance();
		userItemDissimilarityMap = contentAverageDissimilarity.getUserItemAvgDistanceMap(snapshot);
		userThresholdMap = contentAverageDissimilarity.getAverageMap(snapshot, popModel, userItemDissimilarityMap);
	}

	private void trainParameters() {
		LongCollection userIds = snapshot.getUserIds();
		for (long userId : userIds) {
			trainForEachUser(userId);
		}
	}

	private void trainForEachUser(long userId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		AverageAggregate aggregate = userThresholdMap.get(userId);
		WeightAggregate weightAggregate = getWeight(userId);
		for (int i = 0; i < 100; i++) {
			for (IndexedPreference innerPref : prefs) {
				for (IndexedPreference outerPref : prefs) {
					if (innerPref.getItemId() == outerPref.getItemId()) {
						continue;
					}
					Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue(), aggregate);
					Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue(), aggregate);
					double innerSer = getSerendipity(innerTriple, aggregate);
					double outerSer = getSerendipity(outerTriple, aggregate);
					if (innerSer == 0 && outerSer > 0) {
						changeParameters(outerTriple, innerTriple, weightAggregate);
					} else {
						if (innerSer > 0 && outerSer == 0) {
							changeParameters(innerTriple, outerTriple, weightAggregate);
						}
					}
				}
			}
		}
		weightMap.put(userId, weightAggregate);
	}

	private WeightAggregate getWeight(Long userId) {
		if (!weightMap.containsKey(userId)) {
			return new WeightAggregate();
		}
		return weightMap.get(userId);
	}

	private void changeParameters(Triple serTriple, Triple unserTriple, WeightAggregate weight) {
		double pwr = 0, pwd = 0, pwu = 0;
		if (weight.wr + weight.wd + weight.wu > 1) {
			pwr -= pCoeff;
			pwd -= pCoeff;
			pwu -= pCoeff;
		}
		if (weight.wr < 0) {
			pwr += pCoeff;
		}
		if (weight.wd < 0) {
			pwd += pCoeff;
		}
		if (weight.wu < 0) {
			pwu += pCoeff;
		}
		double derR = serTriple.rating - unserTriple.rating;
		double derD = serTriple.dissimilarity - unserTriple.dissimilarity;
		double derU = serTriple.unpopularity - unserTriple.unpopularity;
		weight.wr += learningRate * derR + pwr;
		weight.wd += learningRate * derD + pwd;
		weight.wu += learningRate * derU + pwu;
	}

	private double getPredictedSerendipity(Triple triple, WeightAggregate weight) {
		double result = weight.wr * triple.rating + weight.wd * triple.dissimilarity + weight.wu * triple.unpopularity;
		return result;
	}

	private double getSerendipity(Triple triple, AverageAggregate aggregate) {
		if (triple.rating <= aggregate.getR().getThreshold()) {
			return 0;
		}
		if (triple.dissimilarity <= aggregate.getD().getThreshold()) {
			return 0;
		}
		if (triple.unpopularity <= aggregate.getU().getThreshold()) {
			return 0;
		}
		//double result = triple.rating + triple.dissimilarity + triple.unpopularity;
		return 1;
	}

	private double getDissimilarity(long itemId, long userId) {
		if (!userItemDissimilarityMap.containsKey(userId)) {
			return 1;
		}
		SparseVector vector = userItemDissimilarityMap.get(userId);
		if (!vector.containsKey(itemId)) {
			return 1;
		}
		return vector.get(itemId);
	}

	private void printFunction() {
		LongCollection userIds = snapshot.getUserIds();
		double sum = 0;
		for (long userId : userIds) {
			WeightAggregate weightAggregate = getWeight(userId);
			AverageAggregate aggregate = userThresholdMap.get(userId);
			Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
			for (IndexedPreference innerPref : prefs) {
				for (IndexedPreference outerPref : prefs) {
					if (innerPref.getItemId() == outerPref.getItemId()) {
						continue;
					}
					Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue(), aggregate);
					Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue(), aggregate);
					double innerSer = getSerendipity(innerTriple, aggregate);
					double outerSer = getSerendipity(outerTriple, aggregate);
					double val = 0;
					if (innerSer == 0 && outerSer > 0) {
						val = getPredictedSerendipity(outerTriple, weightAggregate) - getPredictedSerendipity(innerTriple, weightAggregate);
					} else {
						if (innerSer > 0 && outerSer == 0) {
							val = getPredictedSerendipity(innerTriple, weightAggregate) - getPredictedSerendipity(outerTriple, weightAggregate);
						}
					}
					sum += val;
				}
			}
		}
		System.out.println("Function " + sum);
	}

	@Override
	public InvestigationPerUserModel get() {
		System.out.println(InvestigationPerUserModelBuilder.class);
		for (int i = 0; i < iterationCount; i++) {
			System.out.println("Iteration " + i);
			WeightAggregate weightAggregate = getAverageWeights();
			System.out.println("Params wr " + weightAggregate.wr + "; wd " + weightAggregate.wd + "; wu " + weightAggregate.wu);
			printFunction();
			trainParameters();
		}
		print();
		return new InvestigationPerUserModel(0, 0, 0);
	}

	private void print() {
		try {
			PrintWriter writer = new PrintWriter(new File("profile_weights"));
			for (Map.Entry<Long, WeightAggregate> entry : weightMap.entrySet()) {
				int size = snapshot.getUserRatings(entry.getKey()).size();
				writer.println(size + "\t" + entry.getValue().wr + "\t" + entry.getValue().wd + "\t" + entry.getValue().wu);
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private WeightAggregate getAverageWeights() {
		double wr = 0, wd = 0, wu = 0;
		double minr = 0, mind = 0, minu = 0;
		double maxr = 0, maxd = 0, maxu = 0;
		for (WeightAggregate weightAggregate : weightMap.values()) {
			wr += weightAggregate.wr;
			wd += weightAggregate.wd;
			wu += weightAggregate.wu;
			minr = Math.min(minr, weightAggregate.wr);
			mind = Math.min(mind, weightAggregate.wd);
			minu = Math.min(minu, weightAggregate.wu);
			maxr = Math.max(maxr, weightAggregate.wr);
			maxd = Math.max(maxd, weightAggregate.wd);
			maxu = Math.max(maxu, weightAggregate.wu);
		}
		wr /= weightMap.size();
		wd /= weightMap.size();
		wu /= weightMap.size();
		WeightAggregate weightAggregate = new WeightAggregate();
		weightAggregate.wr = wr;
		weightAggregate.wd = wd;
		weightAggregate.wu = wu;
		System.out.println("minr " + minr + " mind " + mind + " minu " + minu);
		System.out.println("maxr " + maxr + " maxd " + maxd + " maxu " + maxu);
		return weightAggregate;
	}

	private class Triple {
		private double rating;
		private double dissimilarity;
		private double unpopularity;

		private Triple(long userId, long itemId, double rating, AverageAggregate aggregate) {
			this.rating = aggregate.getR().getNormalizer().norm(rating);
			dissimilarity = aggregate.getD().getNormalizer().norm(getDissimilarity(itemId, userId));
			unpopularity = aggregate.getU().getNormalizer().norm(1 - (double) popModel.getPop(itemId) / popModel.getMax());
		}
	}

	private class WeightAggregate {
		private double wr = defaultVal;
		private double wd = defaultVal;
		private double wu = defaultVal;
	}
}