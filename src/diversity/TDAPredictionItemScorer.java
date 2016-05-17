package diversity;

import annotation.Alpha;
import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import diversity.genrePredictor.SVDGenreModel;
import evaluationMetric.Container;
import lc.Normalizer;
import mikera.vectorz.AVector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.ContentAverageDissimilarity;

import javax.inject.Inject;
import java.util.*;

public class TDAPredictionItemScorer extends TDAAdvancedItemScorer {
	protected final SVDGenreModel model;
	private Map<Long, Double> importanceMap;

	@Inject
	public TDAPredictionItemScorer(@RatingPredictor ItemScorer itemScorer, PreferenceSnapshot snapshot,
								   @Alpha double factor, @DissimilarityWeight double weight, SVDGenreModel model,
								   PopModel popModel) {
		super(itemScorer, snapshot, factor, weight, popModel);
		this.model = model;
	}

	@Override
	protected void updateGenreSet(long userId) {
		super.updateGenreSet(userId);
		importanceMap = getGenreImportance(userId, genreSet);
	}

	@Override
	protected double getUnexpectedness(Set<Long> genres, Long itemId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		SparseVector vector = map.get(itemId);
		double unexpWeight = super.getUnexpectedness(genres, itemId);
		double maxImportance = 0.0;
		for (Long key : vector.keySet()) {
			if (genreSet.contains(key)) {
				maxImportance = Math.max(importanceMap.get(key), maxImportance);
			}
		}
		return maxImportance + unexpWeight;
	}

	private Map<Long, Double> getGenreImportance(Long userId, Set<Long> genreSet) {
		Map<Long, Double> map = new HashMap<Long, Double>();
		double max = Double.MIN_VALUE, min = Double.MAX_VALUE;
		for (long genreId : genreSet) {
			double val = model.getPrediction(userId, genreId);
			max = Math.max(max, val);
			min = Math.min(min, val);
			map.put(genreId, val);
		}
		Normalizer normalizer = new Normalizer(min, max);
		Iterator<Map.Entry<Long, Double>> iterator = map.entrySet().iterator();
		while (iterator.hasNext()) {
			Map.Entry<Long, Double> entry = iterator.next();
			entry.setValue(normalizer.norm(entry.getValue()));
		}
		return map;
	}
}
