package lc;

public class Normalizer {
	private final double min;
	private final double max;

	public Normalizer(double min, double max) {
		this.min = min;
		this.max = max;
	}

	public double norm(double val) {
		if (min == max) {
			return val / max;
		}
		return (val - min) / (max - min);
	}

	public double getMin() {
		return min;
	}

	public double getMax() {
		return max;
	}
}
