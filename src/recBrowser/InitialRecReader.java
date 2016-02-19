package recBrowser;

import java.util.HashSet;
import java.util.Set;

public class InitialRecReader extends BrowserFileReader {
	private Set<String> algSet;
	private Set<String> userSet;
	private boolean firstLine = true;

	public InitialRecReader() {
		algSet = new HashSet<String>();
		userSet = new HashSet<String>();
	}

	public Set<String> getAlgSet() {
		return algSet;
	}

	public Set<String> getUserSet() {
		return userSet;
	}

	@Override
	protected void processLine(String line) {
		if (firstLine) {
			firstLine = false;
			return;
		}
		String[] brokenLine = line.split(",");
		algSet.add(brokenLine[0]);
		userSet.add(brokenLine[3]);
	}
}
