package lc.advanced;

import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import annotation.RelevanceWeight;
import annotation.UnpopWeight;
import lc.Normalizer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
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
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class LCAdvancedItemScorer extends AbstractItemScorer {
	private PopModel model;
	private final ItemScorer itemScorer;
	private final PreferenceSnapshot snapshot;
	private final LCModel lcModel;
	private final Map<Long, SparseVector> itemContentMap;

	@Inject
	public LCAdvancedItemScorer(PopModel model, @RatingPredictor ItemScorer itemScorer,
								@Transient @Nonnull PreferenceSnapshot snapshot, LCModel lcModel) {
		this.model = model;
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
		this.lcModel = lcModel;
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		itemContentMap = dissimilarity.getItemContentMap();
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		MutableSparseVector ratings = scores.copy();
		itemScorer.score(user, ratings);
		Normalizer ratingNormalizer = Util.getVectorNormalizer(ratings);
		Map<Long, Double> unpopMap = getUnpopMap(scores);
		Normalizer unpopNormalizer = Util.getMapNormalizer(unpopMap);
		Map<Long, Double> dissimMap = getDissimMap(user, scores);
		Normalizer dissimNormalizer = Util.getMapNormalizer(dissimMap);
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = dissimNormalizer.norm(dissimMap.get(e.getKey()));
			double unpop = unpopNormalizer.norm(unpopMap.get(e.getKey()));
			double rating = ratingNormalizer.norm(ratings.get(e.getKey()));
			double total = lcModel.getUw() * unpop + lcModel.getRw() * rating / Settings.MAX + lcModel.getDw() * dissim;
			scores.set(e, total);
		}
	}

	private Map<Long, Double> getDissimMap(long userId, MutableSparseVector scores) {
		Map<Long, Double> dissimMap = new HashMap<Long, Double>();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = getDissim(userId, e.getKey());
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

	private double getDissim(long userId, long itemId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		SparseVector itemVec = itemContentMap.get(itemId);
		double dissim = 0.0;
		for (IndexedPreference p : prefs) {
			SparseVector ratedItemVec = itemContentMap.get(p.getItemId());
			dissim += 1.0 - ContentUtil.getCosine(ratedItemVec, itemVec);
		}
		return dissim / prefs.size();
	}
}
