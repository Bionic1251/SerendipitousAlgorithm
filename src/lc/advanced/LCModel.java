package lc.advanced;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;

@DefaultProvider(LCModelBuilder.class)
@Shareable
public class LCModel {
	private final double rw;
	private final double dw;
	private final double uw;

	public LCModel(double rw, double dw, double uw) {
		this.rw = rw;
		this.dw = dw;
		this.uw = uw;
	}

	public double getRw() {
		return rw;
	}

	public double getDw() {
		return dw;
	}

	public double getUw() {
		return uw;
	}
}
