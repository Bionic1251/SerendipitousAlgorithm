package lc.advanced;

import annotation.Threshold;
import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.AlgorithmUtil;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.Collection;
import java.util.Map;

public class LCModelBuilder implements Provider<LCModel> {
	private final PopModel popModel;
	private final PreferenceSnapshot snapshot;
	private final double threshold;
	private Map<Long, SparseVector> userItemDissimilarityMap;
	private final PreferenceDomain domain;

	private final double dThreshold = 0.2;
	private final double uThreshold = 0.2;

	private final double learningRate = 0.0001;
	private final double regularizationTerm = 0.001;
	private final int iterationCount = 5;

	private double wr = 0.1;
	private double wd = 0.1;
	private double wu = 0.1;

	@Inject
	public LCModelBuilder(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot,
						  @Threshold double threshold, PreferenceDomain domain) {
		this.popModel = popModel;
		this.snapshot = snapshot;
		this.threshold = threshold;
		userItemDissimilarityMap = ContentUtil.getUserItemMapZheng(snapshot, AlgorithmUtil.itemContentMap);
		this.domain = domain;
	}

	private void trainParameters() {
		LongCollection userIds = snapshot.getUserIds();
		for (long userId : userIds) {
			trainForEachUser(userId);
		}
	}

	private void trainForEachUser(long userId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		for (IndexedPreference innerPref : prefs) {
			for (IndexedPreference outerPref : prefs) {
				if (innerPref.getItemId() == outerPref.getItemId()) {
					continue;
				}
				Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue());
				Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue());
				double innerSer = getSerendipity(innerTriple);
				double outerSer = getSerendipity(outerTriple);
				if (innerSer > outerSer) {
					changeParameters(innerTriple, outerTriple);
				} else if (innerSer < outerSer) {
					changeParameters(outerTriple, innerTriple);
				}
			}
		}
	}

	private void changeParameters(Triple serTriple, Triple unserTriple) {
		double derR = (serTriple.rating - unserTriple.rating) / domain.getMaximum();
		double derD = serTriple.dissimilarity - unserTriple.dissimilarity;
		double derU = serTriple.unpopularity - unserTriple.unpopularity;
		wr += learningRate * (derR - regularizationTerm * wr);
		wd += learningRate * (derD - regularizationTerm * wd);
		wu += learningRate * (derU - regularizationTerm * wu);
	}

	private double getPredictedSerendipity(Triple triple) {
		double result = wr * triple.rating / domain.getMaximum() + wd * triple.dissimilarity + wu * triple.unpopularity;
		return result;
	}

	private double getSerendipity(Triple triple) {
		if (triple.rating <= threshold) {
			return 0;
		}
		if (triple.dissimilarity <= dThreshold) {
			return 0;
		}
		if (triple.unpopularity <= uThreshold) {
			return 0;
		}
		double result = triple.rating / domain.getMaximum() + triple.dissimilarity + triple.unpopularity;
		return result;
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
			Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
			for (IndexedPreference innerPref : prefs) {
				for (IndexedPreference outerPref : prefs) {
					if (innerPref.getItemId() == outerPref.getItemId()) {
						continue;
					}
					Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue());
					Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue());
					double innerSer = getSerendipity(innerTriple);
					double outerSer = getSerendipity(outerTriple);
					double val = 0;
					if (innerSer > outerSer) {
						val = getPredictedSerendipity(innerTriple) - getPredictedSerendipity(outerTriple);
					} else if (innerSer < outerSer) {
						val = getPredictedSerendipity(outerTriple) - getPredictedSerendipity(innerTriple);
					}
					sum += val;
				}
			}
		}
		System.out.println("Function " + sum);
	}

	@Override
	public LCModel get() {
		System.out.println(LCModelBuilder.class);
		for (int i = 0; i < iterationCount; i++) {
			System.out.println("Iteration " + i);
			System.out.println("Params wr " + wr + "; wd " + wd + "; wu " + wu);
			printFunction();
			trainParameters();
		}
		return new LCModel(wr, wd, wu);
	}

	private class Triple {
		private double rating;
		private double dissimilarity;
		private double unpopularity;

		private Triple(long userId, long itemId, double rating) {
			this.rating = rating;
			dissimilarity = getDissimilarity(itemId, userId);
			unpopularity = 1 - (double) popModel.getPop(itemId) / popModel.getMax();
		}
	}
}
