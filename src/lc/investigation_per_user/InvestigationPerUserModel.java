package lc.investigation_per_user;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;

@DefaultProvider(InvestigationPerUserModelBuilder.class)
@Shareable
public class InvestigationPerUserModel {
	private final double rw;
	private final double dw;
	private final double uw;

	public InvestigationPerUserModel(double rw, double dw, double uw) {
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
