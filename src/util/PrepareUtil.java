package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.PrintWriter;
import java.util.*;

public class PrepareUtil {
	public static void prepareSmallDataset(String path) {
		int boolLen = 23, boolStart = 5;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("small_content.dat"));
			try {
				String text;
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\\|");
					text = vector[0];
					for (int i = boolStart; i <= boolLen; i++) {
						text += "," + vector[i];
					}
					writer.println(text);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareBigDataset(String path) {
		Map<String, Integer> attributeMap = getAttributesMap(path);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("big_content.dat"));
			try {
				String line = reader.readLine();
				String[] vector = line.split("\t");
				String id = vector[0];
				int[] values = new int[attributeMap.size()];
				while (line != null) {
					vector = line.split("\t");
					String newId = vector[0];
					if (!newId.equals(id)) {
						saveItem(id, values, writer);
						id = newId;
						values = new int[attributeMap.size()];
						continue;
					} else {
						int index = attributeMap.get(vector[1]);
						values[index] = 1;
					}

					line = reader.readLine();
					if (line == null) {
						saveItem(id, values, writer);
					}
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void saveItem(String id, int[] values, PrintWriter writer) {
		String text = id;
		for (int i = 0; i < values.length; i++) {
			text += "," + values[i];
		}
		writer.println(text);
	}

	private static Map<String, Integer> getAttributesMap(String path) {
		Set<String> set = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					set.add(vector[1]);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		List<String> list = new ArrayList<String>(set);
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}
		return map;
	}
}
