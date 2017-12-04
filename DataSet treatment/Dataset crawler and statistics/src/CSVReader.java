import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class CSVReader {
	private File file;
	private BufferedReader reader;
	
	String[] keys;

	public CSVReader(String fileName) {
		try {
			file = new File(fileName);
			reader = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String[] getKeys(){
		return keys;
	}

	public HashMap<String, ArrayList<String>> readCSV(String separator, boolean readHeader) {
		HashMap<String, ArrayList<String>> map = new HashMap<>();

		try {
			String header = reader.readLine().replaceAll("\"", "");
			keys = header.split(separator);
			
			if (readHeader == true) {
				keys = header.split(separator);
				header = reader.readLine().replaceAll("\"", "");;
			}else{
				for(int i = 0; i < keys.length; i++)
					keys[i] = Integer.toString(i);
			}
			
			for(int i = 0; i < keys.length; i++){
				String key = keys[i];
				map.put(key, new ArrayList<String>());
			}
			
			do{
				String[] splited = header.split(separator);
				for(int i = 0; i < keys.length; i++){
					ArrayList<String> values = map.get(keys[i]);
					values.add(splited[i]);
				}
				header = reader.readLine();
				
				if(header!= null)
					header = header.replaceAll("\"", "");
				
			}while(header!=null);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return map;
	}
}
