package adamopoulos;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.Map;

@DefaultProvider(AdaModelBuilder.class)
@Shareable
public class AdaModel {
	private double q = 1;
	private double lambda = -1;
	private double averageDistance;
	private ItemScorer baseline;
	private Map<Long, SparseVector> userItemDistanceMap;

	public AdaModel(double averageDistance, ItemScorer baseline, Map<Long, SparseVector> userItemDistanceMap) {
		this.averageDistance = averageDistance;
		this.baseline = baseline;
		this.userItemDistanceMap = userItemDistanceMap;
	}

	public double getRank(double r, double distance) {
		return q * r - lambda * distance;
	}

	public double getRankById(Long userId, Long itemId) {
		double distance = getDistance(userId, itemId);
		double r = baseline.score(userId, itemId);
		return getRank(r, distance);
	}

	public void setQ(double q) {
		this.q = q;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public double getQ() {
		return q;
	}

	public double getLambda() {
		return lambda;
	}

	public double getDissimilarity(Long userId, Long itemId) {
		if (!userItemDistanceMap.containsKey(userId)) {
			return 1.0;
		}
		SparseVector itemMap = userItemDistanceMap.get(userId);
		if (!itemMap.containsKey(itemId)) {
			return 1.0;
		}
		return itemMap.get(itemId);
	}

	public double getDistance(Long userId, Long itemId) {
		double defaultVal = Math.pow(averageDistance - 1, 2);
		if (!userItemDistanceMap.containsKey(userId)) {
			return defaultVal;
		}
		SparseVector itemMap = userItemDistanceMap.get(userId);
		if (!itemMap.containsKey(itemId)) {
			return defaultVal;
		}
		return Math.pow(averageDistance - itemMap.get(itemId), 2);
	}
}
