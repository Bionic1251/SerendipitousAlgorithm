package util;

import lc.Normalizer;

public class UserStatistics {
	private final double threshold;
	private final Normalizer normalizer;

	public UserStatistics(double threshold, Normalizer normalizer) {
		this.normalizer = normalizer;
		this.threshold = normalizer.norm(threshold);
	}

	public double getThreshold() {
		return threshold;
	}

	public Normalizer getNormalizer() {
		return normalizer;
	}
}
