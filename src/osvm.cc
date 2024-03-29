/**************************************************************************
 * This file is part of osvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Gabriella Melki (melkiga@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

#include "osvm.h"

// compile with: g++/gcc/visual studio 
extern "C" { FILE __iob_func[3] = { *stdin,*stdout,*stderr }; }

ostream& operator<< (ostream& os, bopt::variables_map& vars) {
	map<string, bopt::variable_value>::iterator it;
	for (it = vars.begin(); it != vars.end(); it++) {
		if (typeid(string) == it->second.value().type()) {
			os << format("%-20s%s\n") % it->first % it->second.as<string>();
		} else if (typeid(double) == it->second.value().type()) {
			os << format("%-20s%g\n") % it->first % it->second.as<double>();
		} else if (typeid(float) == it->second.value().type()) {
			os << format("%-20s%g\n") % it->first % it->second.as<float>();
		} else if (typeid(long) == it->second.value().type()) {
			os << format("%-20s%l\n") % it->first % it->second.as<long>();
		} else if (typeid(int) == it->second.value().type()) {
			os << format("%-20s%d\n") % it->first % it->second.as<int>();
		} else {
			os << it->first << "\n";
		}
	}
	os << endl;
	return os;
}

int main(int argc, char *argv[]) {
	string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
	string descr = "Perform SVM training for the given data set [FILE].\n";
	string options = "Available options";
	bopt::options_description desc(usage + descr + options);
	desc.add_options()
		(PR_HELP, "produce help message")
		(PR_C_LOW, bopt::value<fvalue>()->default_value(0.001), "C value (lower bound)")
		(PR_C_HIGH, bopt::value<fvalue>()->default_value(10000.0), "C value (upper bound)")
		(PR_G_LOW, bopt::value<fvalue>()->default_value(0.0009765625), "gamma value (lower bound)")
		(PR_G_HIGH, bopt::value<fvalue>()->default_value(16.0), "gamma value (upper bound)")
		(PR_RES, bopt::value<int>()->default_value(8), "resolution (for C and gamma)")
		(PR_OUTER_FLD, bopt::value<int>()->default_value(1), "outer folds")
		(PR_INNER_FLD, bopt::value<int>()->default_value(10), "inner folds")
		(PR_BIAS_CALCULATION, bopt::value<string>()->default_value(BIAS_CALCULATION_YES), "bias evaluation strategy (yes, no)")
		(PR_CREATE_TESTS, bopt::value<bool>()->default_value(false), "create test cases")
		(PR_TEST_NAME, bopt::value<string>()->default_value("test/examples/example.json"), "test case file name (JSON)")
		(PR_CACHE_SIZE, bopt::value<int>()->default_value(DEFAULT_CACHE_SIZE), "cache size (in MB)")
		(PR_EPOCH, bopt::value<fvalue>()->default_value(0.5), "epochs number")
		(PR_MARGIN, bopt::value<fvalue>()->default_value(0.1), "margin")
		(PR_INPUT, bopt::value<string>(), "input file");

	bopt::positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);

	try {	
		bopt::variables_map vars;
		bopt::store(bopt::command_line_parser(argc, argv).options(desc).positional(opt).run(), vars);
		bopt::notify(vars);

		if (!vars.count(PR_KEY_HELP)) {
			ParametersParser parser(vars);
			Configuration conf = parser.getConfiguration();

			// logger << vars;

			ApplicationLauncher launcher(conf);
			pt::ptree model_tree;
			model_tree.put_child("config", pt::ptree());
			model_tree.put_child("classifier", pt::ptree());
			launcher.run(model_tree);
			pt::write_json(cout, model_tree.get_child("classifier"));

			pt::ptree model_tree_two;
			model_tree_two.put_child("config", pt::ptree());
			model_tree_two.put_child("classifier", pt::ptree());
			launcher.run(model_tree_two);
			pt::write_json(cout, model_tree_two.get_child("classifier"));

			// create test cases if user specified
			if(conf.createTestCases){
				pt::write_json(conf.testName, model_tree);	
			}
		} else {
			cerr << desc;
		}
	} catch (exception& e) {
		cerr << "\033[1;31m Error: \033[0m " << e.what() << "\n" << endl;
		cerr << desc;
	}

	return 0;
}
