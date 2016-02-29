package lc.advanced;

import annotation.D_Threshold;
import annotation.R_Threshold;
import annotation.RatingPredictor;
import annotation.U_Threshold;
import it.unimi.dsi.fastutil.longs.LongCollection;
import lc.Normalizer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.*;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class LCModelBuilder implements Provider<LCModel> {
	private final PopModel popModel;
	private final PreferenceSnapshot snapshot;
	private final double rThreshold;
	private final double dThreshold;
	private final double uThreshold;
	private final ItemScorer itemScorer;
	private Map<Long, SparseVector> userItemDissimilarityMap;

	private final double learningRate = 0.00001;
	private final double regularizationTerm = 0.0001;
	private final int iterationCount = 5;

	private double wr = 0.1;
	private double wd = 0.1;
	private double wu = 0.1;

	@Inject
	public LCModelBuilder(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot, @RatingPredictor ItemScorer itemScorer,
						  @R_Threshold double rThreshold, @D_Threshold double dThreshold, @U_Threshold double uThreshold) {
		this.popModel = popModel;
		this.snapshot = snapshot;
		this.rThreshold = rThreshold;
		this.dThreshold = dThreshold;
		this.uThreshold = uThreshold;
		this.itemScorer = itemScorer;
		ContentAverageDissimilarity contentAverageDissimilarity = ContentAverageDissimilarity.getInstance();
		userItemDissimilarityMap = contentAverageDissimilarity.getUserItemAvgDistanceMap(snapshot);
	}

	private void trainParameters() {
		LongCollection userIds = snapshot.getUserIds();
		for (long userId : userIds) {
			trainForEachUser(userId);
		}
	}

	private void trainForEachUser(long userId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		List<Long> ids = prefsToIds(prefs);
		SparseVector predVector = itemScorer.score(userId, ids);
		Normalizer ratingNormalizer = Util.getVectorNormalizer(predVector);
		Map<Long, Double> dMap = getDissimMap(userId, predVector);
		Normalizer dNormalizer = Util.getMapNormalizer(dMap);
		Map<Long, Double> uMap = getUnpopMap(predVector);
		Normalizer uNormalizer = Util.getMapNormalizer(uMap);
		//Fixme: scale ratings, dissims and unpops!
		for (IndexedPreference innerPref : prefs) {
			for (IndexedPreference outerPref : prefs) {
				if (innerPref.getItemId() == outerPref.getItemId()) {
					continue;
				}
				Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue());
				Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue());
				double innerSer = getSerendipity(innerTriple);
				double outerSer = getSerendipity(outerTriple);
				Triple innerPredTriple = new Triple(userId, innerPref.getItemId(), predVector.get(innerPref.getItemId()));
				innerPredTriple.rating = ratingNormalizer.norm(predVector.get(innerPref.getItemId()));
				innerPredTriple.unpopularity = uNormalizer.norm(uMap.get(innerPref.getItemId()));
				innerPredTriple.dissimilarity = dNormalizer.norm(dMap.get(innerPref.getItemId()));
				Triple outerPredTriple = new Triple(userId, outerPref.getItemId(), predVector.get(outerPref.getItemId()));
				outerPredTriple.rating = ratingNormalizer.norm(predVector.get(outerPref.getItemId()));
				outerPredTriple.dissimilarity = dNormalizer.norm(dMap.get(outerPref.getItemId()));
				outerPredTriple.unpopularity = uNormalizer.norm(uMap.get(outerPref.getItemId()));
				if (innerSer > outerSer) {
					changeParameters(innerPredTriple, outerPredTriple);
				} else if (innerSer < outerSer) {
					changeParameters(outerPredTriple, innerPredTriple);
				}
			}
		}
	}

	private Map<Long, Double> getDissimMap(long userId, SparseVector scores) {
		Map<Long, Double> dissimMap = new HashMap<Long, Double>();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = getDissimilarity(userId, e.getKey());
			dissimMap.put(e.getKey(), dissim);
		}
		return dissimMap;
	}

	private Map<Long, Double> getUnpopMap(SparseVector scores) {
		Map<Long, Double> unpopMap = new HashMap<Long, Double>();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double unpop = 1.0 - (double) popModel.getPop(e.getKey()) / popModel.getMax();
			unpopMap.put(e.getKey(), unpop);
		}
		return unpopMap;
	}

	private List<Long> prefsToIds(Collection<IndexedPreference> prefs) {
		List<Long> ids = new ArrayList<Long>();
		for (IndexedPreference preference : prefs) {
			ids.add(preference.getItemId());
		}
		return ids;
	}

	private void changeParameters(Triple serTriple, Triple unserTriple) {
		double derR = (serTriple.rating - unserTriple.rating) / Settings.MAX;
		double derD = serTriple.dissimilarity - unserTriple.dissimilarity;
		double derU = serTriple.unpopularity - unserTriple.unpopularity;
		wr += learningRate * (derR - regularizationTerm * wr);
		wd += learningRate * (derD - regularizationTerm * wd);
		wu += learningRate * (derU - regularizationTerm * wu);
	}

	private double getPredictedSerendipity(Triple triple) {
		double result = wr * triple.rating / Settings.MAX + wd * triple.dissimilarity + wu * triple.unpopularity;
		return result;
	}

	private double getSerendipity(Triple triple) {
		if (triple.rating <= rThreshold) {
			return 0;
		}
		if (triple.dissimilarity <= dThreshold) {
			return 0;
		}
		if (triple.unpopularity <= uThreshold) {
			return 0;
		}
		double result = triple.rating / Settings.MAX + triple.dissimilarity + triple.unpopularity;
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
			List<Long> ids = prefsToIds(prefs);
			SparseVector predVector = itemScorer.score(userId, ids);
			Normalizer ratingNormalizer = Util.getVectorNormalizer(predVector);
			Map<Long, Double> dMap = getDissimMap(userId, predVector);
			Normalizer dNormalizer = Util.getMapNormalizer(dMap);
			Map<Long, Double> uMap = getUnpopMap(predVector);
			Normalizer uNormalizer = Util.getMapNormalizer(uMap);
			for (IndexedPreference innerPref : prefs) {
				for (IndexedPreference outerPref : prefs) {
					if (innerPref.getItemId() == outerPref.getItemId()) {
						continue;
					}
					Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue());
					Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue());
					double innerSer = getSerendipity(innerTriple);
					double outerSer = getSerendipity(outerTriple);
					Triple innerPredTriple = new Triple(userId, innerPref.getItemId(), predVector.get(innerPref.getItemId()));
					innerPredTriple.rating = ratingNormalizer.norm(predVector.get(innerPref.getItemId()));
					innerPredTriple.unpopularity = uNormalizer.norm(uMap.get(innerPref.getItemId()));
					innerPredTriple.dissimilarity = dNormalizer.norm(dMap.get(innerPref.getItemId()));
					Triple outerPredTriple = new Triple(userId, outerPref.getItemId(), predVector.get(outerPref.getItemId()));
					outerPredTriple.rating = ratingNormalizer.norm(predVector.get(outerPref.getItemId()));
					outerPredTriple.dissimilarity = dNormalizer.norm(dMap.get(outerPref.getItemId()));
					outerPredTriple.unpopularity = uNormalizer.norm(uMap.get(outerPref.getItemId()));
					double val = 0;
					if (innerSer > outerSer) {
						val = getPredictedSerendipity(innerPredTriple) - getPredictedSerendipity(outerPredTriple);
					} else if (innerSer < outerSer) {
						val = getPredictedSerendipity(outerPredTriple) - getPredictedSerendipity(innerPredTriple);
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