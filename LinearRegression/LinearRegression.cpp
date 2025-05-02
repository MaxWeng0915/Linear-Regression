#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>

#include <ctime>

using namespace std;


struct DataRecord {
	double YearsExperience;
	double EducationLevel;
	double CityA;
	double CityB;
	double Salary;
};

int loadcsvfile2(string fname, vector<DataRecord>& data) {
	ifstream inputFile(fname);
	if (!inputFile.is_open()) {
		cerr << "Could not open the file" << endl;
		return 0;
	}

	 string line;

	if (getline(inputFile, line)) {
		while (getline(inputFile, line)) {
			stringstream ss(line);
			DataRecord record;
			char comma;
			ss >> record.YearsExperience >> comma;
			string _EducationLevel;
			getline(ss, _EducationLevel, ',');
			if("master" == _EducationLevel)
				record.EducationLevel = 2;
			else if ("university" == _EducationLevel)
				record.EducationLevel = 1;
			else if ("highschool" == _EducationLevel)
				record.EducationLevel = 0;
			string _City;
			getline(ss, _City, ',');
			if ("CityA" == _City) {
				record.CityA = 1;
				record.CityB = 0;
			}
			else if ("CityB" == _City) {
				record.CityA = 0;
				record.CityB = 1;
			}
			else if ("CityC" == _City) {
				//what if the city is c, means not city a and city b

				record.CityA = 0;
				record.CityB = 0;
			}
			ss >> record.Salary;
			data.push_back(record);
		}
	}

	inputFile.close();
	return 1;
}

float compute_cost(vector<double> _x, vector<double> _y, float w, float b) {
	float sum = 0.0;
	for (int i = 0; i < _x.size(); i++) {

		float y = _y[i];
		float x = _x[i];
		float y_pred = w * x + b;
		float cost = (y - y_pred) * (y - y_pred);
		sum += cost;
	}
	return sum / _x.size();
}

void compute_gradient(vector<double> _x, vector<double> _y, float w, float b, float& _w_gradient, float& _b_gradient) {
	float sumw = 0.0;
	float sumb = 0.0;
	for (int i = 0; i < _x.size(); i++) {

		float y = _y[i];
		float x = _x[i];
		float w_gradient = x * (w * x + b - y);
		float b_gradient = (w * x + b - y);
		sumw += w_gradient;
		sumb += b_gradient;
	}
	_w_gradient = sumw / _x.size();
	_b_gradient = sumb / _x.size();
}

float compute_cost(vector<vector<double>> _x, vector<double> _y, vector<float> w, float b) {
	int size = w.size();

	float sum = 0.0;
	for (int i = 0; i < _y.size(); i++) {

		float y = _y[i];
		float y_pred = b;
		for (int j = 0; j < size; j++) {
			y_pred += w[j] * _x[i][j];
		}
		float cost = (y - y_pred) * (y - y_pred);
		sum += cost;		
	}
	return sum / _y.size();
}

void compute_gradient(vector<vector<double>> _x, vector<double> _y, vector<float> w, float b, vector<float>& _w_gradient, float& _b_gradient) {
	int size = w.size();
	vector<float> sumw(size, 0.0);
	float sumb = 0.0;
	for (int i = 0; i < _x.size(); i++) {

		float y = _y[i];
		vector<float> w_gradients(size, 0.0);
		
		// y_pred = w1 * x1 + ... + wn *xn + b
		float y_pred = b;
		for (int j = 0; j < size; j++) {
			y_pred += w[j] * _x[i][j];
		}

		for (int j = 0; j < size; j++) {
			w_gradients[j] = _x[i][j] * (y_pred - y);
			sumw[j] += w_gradients[j];
		}	
		float b_gradient = (y_pred - y);
		sumb += b_gradient;		
	}

	for (int j = 0; j < size; j++)
		_w_gradient[j] = sumw[j] / _x.size();
	_b_gradient = sumb / _x.size();
}

double calculateMean(const vector<double>& data) {
	double sum = 0.0;
	for (double value : data) {
		sum += value;
	}
	return sum / data.size();
}

double calculateStandardDeviation(const vector<double>& data) {
	double mean = calculateMean(data);
	double sumOfSquaredDifferences = 0.0;

	for (double value : data) {
		double difference = value - mean;
		sumOfSquaredDifferences += difference * difference;
	}

	double variance = sumOfSquaredDifferences / data.size();
	double standardDeviation = sqrt(variance);

	return standardDeviation;
}


vector<double> featureScaling(vector<double> data) {	
	double mean = calculateMean(data);
	double sd = calculateStandardDeviation(data);
	for(int i = 0; i<data.size();i++)
		data[i] = (data[i] - mean) / sd;

	return data;

}

void splitdata(vector<vector<double>> datas, vector<vector<double>>& train, vector<vector<double>>& test, float percent = 0.2) {
	int len = datas.size();
	int train_len = len * (1.0 - percent);
	int test_len = len  - train_len;

	int n = 0;

	std::srand(std::time(nullptr));
	std::vector<int> numbers(len, 0);

	while (train_len > n) {
		int random_number = std::rand() % len;

		if (numbers[random_number] == 0) {
			numbers[random_number] = 1;
			n++;
		}
	}

	for (int i = 0; i < len; i++) {
		if(numbers[i] == 1)
			train.push_back(datas[i]);
		else if (numbers[i] == 0)
			test.push_back(datas[i]);
	}
}
void splitdata(vector<double> datas, vector<double>& train, vector<double>& test, float percent = 0.2) {
	int len = datas.size();
	int train_len = len * (1.0 - percent);
	int test_len = len - train_len;

	int n = 0;

	std::srand(std::time(nullptr));
	std::vector<int> numbers(len, 0);

	while (train_len > n) {
		int random_number = std::rand() % len;

		if (numbers[random_number] == 0) {
			numbers[random_number] = 1;
			n++;
		}		
	}

	for (int i = 0; i < len; i++) {
		if (numbers[i] == 1)
			train.push_back(datas[i]);
		else if (numbers[i] == 0)
			test.push_back(datas[i]);
	}
}

int main()
{

	vector<DataRecord> data;
	map<string, vector<double>> datas;
	int res = loadcsvfile2("Salary_Data2.csv", data);
	if (res == 1) {
		datas["YearsExperience"] = vector<double>();
		datas["EducationLevel"] = vector<double>();
		datas["CityA"] = vector<double>();
		datas["CityB"] = vector<double>();
		datas["Salary"] = vector<double>();

		for (const DataRecord& record : data) {
			datas["YearsExperience"].push_back(record.YearsExperience);
			datas["EducationLevel"].push_back(record.EducationLevel);
			datas["CityA"].push_back(record.CityA);
			datas["CityB"].push_back(record.CityB);
			datas["Salary"].push_back(record.Salary);
		}

		// feature Scaling
		datas["YearsExperience"] = featureScaling(datas["YearsExperience"]);
		datas["EducationLevel"] = featureScaling(datas["EducationLevel"]);
		datas["CityA"] = featureScaling(datas["YearsExperience"]);
		datas["CityB"] = featureScaling(datas["CityB"]);

		vector<float> w = {1, 2, 3, 4};
		float b = 1;
		vector<vector<double>> x;
		for (int i = 0; i < datas["YearsExperience"].size(); i++)
		{
			vector<double> x_sub;
			x_sub.push_back(datas["YearsExperience"][i]);
			x_sub.push_back(datas["EducationLevel"][i]);
			x_sub.push_back(datas["CityA"][i]);
			x_sub.push_back(datas["CityB"][i]);
			x.push_back(x_sub);
		}		
		vector<vector<double>> x_train, x_test;
		vector<double> y_train, y_test;

		splitdata(x, x_train, x_test);
		splitdata(datas["Salary"], y_train, y_test);

	
		float learning_rate = 0.001;
		float cost;
		
		for (int i = 0; i < 20000; i++) {
			vector<float> w_gradient = { 0.0, 0.0, 0.0, 0.0 };
			float b_gradient;
			compute_gradient(x_train, y_train, w, b, w_gradient, b_gradient);

			w[0] = w[0] - w_gradient[0] * learning_rate;
			w[1] = w[1] - w_gradient[1] * learning_rate;
			w[2] = w[2] - w_gradient[2] * learning_rate;
			w[3] = w[3] - w_gradient[3] * learning_rate;
			b = b - b_gradient * learning_rate;

			cost = compute_cost(x, datas["Salary"], w, b);

			if (i % 1000 == 0)
				cout << "i: " << i << " Cost: " << cost << " w[0]:" << w[0] << " b:" << b << " w_gradient[0]:" << w_gradient[0] << " b_gradient:" << b_gradient << endl;
		}

		cout << "================ test data ================" << endl;	

		for (int i = 0; i < x_test.size(); i++) {
			float y = y_test[i];
			float y_pred = b;
			for (int j = 0; j < w.size(); j++) {
				y_pred += w[j] * x_test[i][j];
			}

			cout << "y_pred: " << y_pred << ", y: " << y << endl;
		}

	}

	std::cin.get();
	return 0;

}