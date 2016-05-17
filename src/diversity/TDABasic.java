package diversity;

import annotation.RatingPredictor;
import evaluationMetric.Container;
import lc.Normalizer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import util.ContentAverageDissimilarity;
import util.ContentUtil;
import util.Settings;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

public abstract class TDABasic extends AbstractItemScorer {
	protected final ItemScorer itemScorer;
	protected static final int CANDIDATE_NUMBER = 50;
	protected final PreferenceSnapshot snapshot;

	public TDABasic(ItemScorer itemScorer, PreferenceSnapshot snapshot) {
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		MutableSparseVector predictedScores = scores.copy();
		itemScorer.score(user, predictedScores);
		List<Container<Double>> candidateList = getCandidates(user, predictedScores);
		List<Long> diversified = diversify(candidateList, user);
		for (Long key : predictedScores.keySet()) {
			if (diversified.indexOf(key) == -1) {
				scores.set(key, 0.0);
			} else {
				int score = diversified.size() - diversified.indexOf(key);
				scores.set(key, score);
			}
		}
	}

	protected List<Long> diversify(List<Container<Double>> candidateList, long userId) {
		List<Long> diversifiedList = new ArrayList<Long>();
		diversifiedList.add(candidateList.get(0).getId());
		for (int i = 1; i < candidateList.size(); i++) {
			List<Container<Double>> newCandidateList = getCutCandidateList(diversifiedList, candidateList);
			List<Container<Double>> sortedList = sortCandidates(diversifiedList, newCandidateList);
			List<Container<Double>> result = mixLists(newCandidateList, sortedList, userId);
			diversifiedList.add(result.get(0).getId());
		}
		return diversifiedList;
	}

	protected abstract List<Container<Double>> mixLists(List<Container<Double>> relevantCandidates, List<Container<Double>> diversifiedCandidates, long userId);

	private List<Container<Double>> sortCandidates(List<Long> diversified, List<Container<Double>> candidate) {
		List<Container<Double>> containerList = new ArrayList<Container<Double>>();
		for (Container<Double> item : candidate) {
			containerList.add(new Container<Double>(item.getId(), getDissimilarity(diversified, item)));
		}
		Collections.sort(containerList);
		Collections.reverse(containerList);
		return containerList;
	}

	private double getDissimilarity(List<Long> diversified, Container<Double> item) {
		return 1.0 - getSimilarity(diversified, item);
	}

	private double getSimilarity(List<Long> diversified, Container<Double> item) {
		Long itemId = item.getId();
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		SparseVector itemVector = map.get(itemId);
		double sim = 0;
		for (Long selectedItemId : diversified) {
			SparseVector selectedVector = map.get(selectedItemId);
			sim += ContentUtil.getJaccard(itemVector, selectedVector);
		}
		return sim / diversified.size();
	}

	private List<Container<Double>> getCutCandidateList(List<Long> diversified, List<Container<Double>> candidate) {
		List<Container<Double>> newCandidate = new ArrayList<Container<Double>>(candidate);
		Iterator<Container<Double>> iterator = newCandidate.iterator();
		while (iterator.hasNext()) {
			Container<Double> item = iterator.next();
			if (diversified.indexOf(item.getId()) != -1) {
				iterator.remove();
			}
		}
		return newCandidate;
	}

	private List<Container<Double>> getCandidates(long userId, MutableSparseVector scores) {
		Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
		Set<Long> ratedSet = ratingsToSet(ratings);
		List<Container<Double>> predictedList = getOrderedPredictedList(scores);
		List<Container<Double>> candidates = new ArrayList<Container<Double>>();
		for (Container<Double> container : predictedList) {
			if (candidates.size() >= CANDIDATE_NUMBER) {
				break;
			}
			if (!ratedSet.contains(container.getId())) {
				candidates.add(container);
			}
		}
		return candidates;
	}

	private List<Container<Double>> getOrderedPredictedList(MutableSparseVector scores) {
		List<Container<Double>> scoreList = new ArrayList<Container<Double>>();
		for (Long key : scores.keySet()) {
			scoreList.add(new Container<Double>(key, scores.get(key)));
		}
		Collections.sort(scoreList);
		Collections.reverse(scoreList);
		return scoreList;
	}

	private Set<Long> ratingsToSet(Collection<IndexedPreference> ratings) {
		Set<Long> set = new HashSet<Long>();
		for (IndexedPreference pref : ratings) {
			set.add(pref.getItemId());
		}
		return set;
	}
}
