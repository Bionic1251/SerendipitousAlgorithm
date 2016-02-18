package lc;

public class Normalizer {
	private final double min;
	private final double max;

	public Normalizer(double min, double max) {
		this.min = min;
		this.max = max;
	}

	public double norm(double val) {
		return (val - min) / (max - min);
	}
}
