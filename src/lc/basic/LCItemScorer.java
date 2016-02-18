package lc.basic;

import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import annotation.RelevanceWeight;
import annotation.UnpopWeight;
import lc.Normalizer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.AlgorithmUtil;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class LCItemScorer extends AbstractItemScorer {
	private PopModel model;
	private final ItemScorer itemScorer;
	private final PreferenceSnapshot snapshot;
	private final double dissimilarityWeight;
	private final double unpopWeight;
	private final double relevanceWeight;
	private final PreferenceDomain domain;

	@Inject
	public LCItemScorer(PopModel model, @RatingPredictor ItemScorer itemScorer,
						@Transient @Nonnull PreferenceSnapshot snapshot, @DissimilarityWeight double dissimilarity,
						@UnpopWeight double unpop, @RelevanceWeight double relevance, PreferenceDomain domain) {
		System.out.println("LC R " + relevance + "; D " + dissimilarity + "; U " + unpop);
		this.model = model;
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
		dissimilarityWeight = dissimilarity;
		unpopWeight = unpop;
		relevanceWeight = relevance;
		this.domain = domain;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		MutableSparseVector ratings = scores.copy();
		itemScorer.score(user, ratings);
		Normalizer ratingNormalizer = getRatingNormalizer(ratings);
		Map<Long, Double> unpopMap = getUnpopMap(scores);
		Normalizer unpopNormalizer = getMapNormalizer(unpopMap);
		Map<Long, Double> dissimMap = getDissimMap(user, scores);
		Normalizer dissimNormalizer = getMapNormalizer(dissimMap);
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = dissimNormalizer.norm(dissimMap.get(e.getKey()));
			double unpop = unpopNormalizer.norm(unpopMap.get(e.getKey()));
			double rating = ratingNormalizer.norm(ratings.get(e.getKey()));
			double total = unpopWeight * unpop + relevanceWeight * rating / domain.getMaximum() + dissimilarityWeight * dissim;
			scores.set(e, total);
		}
	}

	private Normalizer getMapNormalizer(Map<Long, Double> map) {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (double val : map.values()) {
			min = Math.min(min, val);
			max = Math.max(max, val);
		}
		return new Normalizer(min, max);
	}

	private Map<Long, Double> getDissimMap(long userId, MutableSparseVector scores) {
		Map<Long, Double> dissimMap = new HashMap<Long, Double>();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = 0;
			if (dissimilarityWeight != 0.0) {
				dissim = getDissim(userId, e.getKey());
			}
			dissimMap.put(e.getKey(), dissim);
		}
		return dissimMap;
	}

	private Map<Long, Double> getUnpopMap(MutableSparseVector scores) {
		Map<Long, Double> unpopMap = new HashMap<Long, Double>();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double unpop = 1.0 - (double) model.getPop(e.getKey()) / model.getMax();
			unpopMap.put(e.getKey(), unpop);
		}
		return unpopMap;
	}

	private Normalizer getRatingNormalizer(MutableSparseVector ratings) {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (VectorEntry e : ratings.view(VectorEntry.State.EITHER)) {
			min = Math.min(min, e.getValue());
			max = Math.max(max, e.getValue());
		}
		return new Normalizer(min, max);
	}

	private double getDissim(long userId, long itemId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		SparseVector itemVec = AlgorithmUtil.itemContentMap.get(itemId);
		double dissim = 0.0;
		for (IndexedPreference p : prefs) {
			SparseVector ratedItemVec = AlgorithmUtil.itemContentMap.get(p.getItemId());
			dissim += 1.0 - ContentUtil.getCosine(ratedItemVec, itemVec);
		}
		return dissim / prefs.size();
	}
}