import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

public class Test {
    public static void main(String[] args) throws IOException {
        CsvReader csvReader1 = new CsvReader("D:\\My_code\\python\\learning-records-main\\code\\travel_time_estimation\\data\\road_data\\link1.csv", ',', Charset.forName("UTF-8"));
        csvReader1.readHeaders();

        String outPath = "D:\\My_code\\python\\learning-records-main\\code\\travel_time_estimation\\data\\road_data\\";
        String outFile = outPath + "\\" + "result.csv";
        CsvWriter csvwriter = null;
        File csvFile = new File(outFile);

        if (csvFile.createNewFile())
            System.out.println("create file");

        csvwriter = new CsvWriter(csvFile.getCanonicalPath(), ',', Charset.forName("UTF-8"));
        String[] ss = new String[1];
        ss[0] = "from_id,to_id,node_id";
        csvwriter.writeRecord(ss, true);

        int i = 0;
        ArrayList<String> result = new ArrayList<String>();
        while (csvReader1.readRecord()){
            CsvReader csvReader2 = new CsvReader("D:\\My_code\\python\\learning-records-main\\code\\travel_time_estimation\\data\\road_data\\link1.csv", ',', Charset.forName("UTF-8"));
            csvReader2.readHeaders();
            while (csvReader2.readRecord()){
                if (Integer.parseInt(csvReader1.get("to_node_id")) == Integer.parseInt(csvReader2.get("from_node_id"))) {
                    String temp = "";
                    temp += csvReader1.get("link_id");
                    temp += ",";
                    temp += csvReader2.get("link_id");
                    temp += ",";
                    temp += csvReader1.get("to_node_id");
                    result.add(temp);
                }
            }

            for (String t : result){
                ss[0] = t;
                csvwriter.writeRecord(ss, true);
            }
            result.clear();

            System.out.println("finish one: " + i++);
//            if (i == 15)
//                break;
        }
    csvwriter.close();
    }
}
