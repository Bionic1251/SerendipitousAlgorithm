package diversity;

import annotation.DissimilarityWeight;
import org.grouplens.lenskit.core.Shareable;

import javax.inject.Inject;

public class SimpleWeight implements TDAWeight {
	private double weight;

	public SimpleWeight(@DissimilarityWeight double weight) {
		this.weight = weight;
	}

	@Override
	public double getWeight(int num) {
		return weight;
	}
}
