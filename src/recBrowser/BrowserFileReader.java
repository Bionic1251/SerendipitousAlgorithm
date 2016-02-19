package recBrowser;

import java.io.BufferedReader;
import java.util.*;

public abstract class BrowserFileReader {
	public void readFile(String filePath) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(filePath));
			try {
				String line = reader.readLine();
				while (line != null) {
					processLine(line);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected abstract void processLine(String line);
}
