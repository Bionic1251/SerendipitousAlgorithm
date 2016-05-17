package diversity;

import annotation.Alpha;
import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import evaluationMetric.Container;
import lc.Normalizer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.ContentAverageDissimilarity;
import util.Settings;
import util.Util;

import javax.inject.Inject;
import java.util.*;

public class TDAAdvancedItemScorer extends TDABasic {
	private double factor;
	protected final double serWeight;
	protected Set<Long> genreSet;
	private final PopModel popModel;

	@Inject
	public TDAAdvancedItemScorer(@RatingPredictor ItemScorer itemScorer, PreferenceSnapshot snapshot,
								 @Alpha double factor, @DissimilarityWeight double weight, PopModel popModel) {
		super(itemScorer, snapshot);
		this.factor = factor;
		serWeight = weight;
		this.popModel = popModel;
		System.out.println("TDAAdvancedItemScorer factor=" + factor + " weightCalc=" + weight);
	}

	@Override
	protected List<Long> diversify(List<Container<Double>> candidateList, long userId) {
		updateGenreSet(userId);
		return super.diversify(candidateList, userId);
	}

	protected void updateGenreSet(long userId) {
		SparseVector userVector = getUserVector(userId);
		genreSet = Util.getGenresByUserVector(userVector);
	}

	protected double getPop(Long itemId) {
		return popModel.getPop(itemId) / popModel.getMax();
	}

	@Override
	protected List<Container<Double>> mixLists(List<Container<Double>> relevantCandidates, List<Container<Double>> diversifiedCandidates, long userId) {
		List<Container<Double>> scoreList = new ArrayList<Container<Double>>();
		updateDiversifiedCandidates(diversifiedCandidates);
		Normalizer relevantNormalizer = getNormalizer(relevantCandidates);
		Normalizer diversifiedNormalizer = getNormalizer(diversifiedCandidates);
		for (Container<Double> item : relevantCandidates) {
			double relevance = relevantNormalizer.norm(item.getValue());
			Container<Double> diversifiedItem = getItemById(diversifiedCandidates, item.getId());
			double dissimilarity = diversifiedNormalizer.norm(diversifiedItem.getValue());
			double score = relevance * (1.0 - factor) + dissimilarity * factor;
			scoreList.add(new Container<Double>(item.getId(), score));
		}
		Collections.sort(scoreList);
		Collections.reverse(scoreList);
		return scoreList;
	}


	protected void updateDiversifiedCandidates(List<Container<Double>> diversifiedCandidates) {
		for (Container<Double> item : diversifiedCandidates) {
			double weight = getUnexpectedness(genreSet, item.getId());
			double newVal = item.getValue() + serWeight * weight;
			item.setValue(newVal);
		}
	}

	protected double getUnexpectedness(Set<Long> genres, Long itemId) {
		return Util.getUnexpectedness(genres, itemId);
	}

	protected SparseVector getUserVector(long userId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		MutableSparseVector vector = dissimilarity.getEmptyVector();
		Collection<IndexedPreference> preferenceCollection = snapshot.getUserRatings(userId);
		for (IndexedPreference pref : preferenceCollection) {
			SparseVector itemVector = map.get(pref.getItemId());
			itemVector = dissimilarity.toTFIDF(itemVector);
			vector.add(itemVector);
		}
		return vector;
	}

	private Container<Double> getItemById(List<Container<Double>> list, Long id) {
		for (Container<Double> container : list) {
			if (container.getId().equals(id)) {
				return container;
			}
		}
		return null;
	}

	private Normalizer getNormalizer(List<Container<Double>> list) {
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
		for (Container<Double> container : list) {
			min = Math.min(min, container.getValue());
			max = Math.max(max, container.getValue());
		}
		return new Normalizer(min, max);
	}
}
