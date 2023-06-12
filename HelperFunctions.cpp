#include <unordered_map>
#include <unordered_set>
#include "HelperFunctions.h"


void ltrim(std::string &_str, const std::string &_chars = "\t\n\v\f\r ") {
    _str.erase(0, _str.find_first_not_of(_chars));
}

void rtrim(std::string &_str, const std::string &_chars = "\t\n\v\f\r ") {
    _str.erase(_str.find_last_not_of(_chars) + 1);
}

void trim(std::string &_str, const std::string &_chars = "\t\n\v\f\r ") {
    ltrim(_str, _chars);
    rtrim(_str, _chars);
}

/*
 *  Get the index of next unblank char from a string.
 */
unsigned int getNextChar(const char *str) {
    unsigned int rtn = 0;
    // Jump over all blanks
    for (; str[rtn] == ' '; ++rtn);
    return rtn;
}

/*
 *  Get next word from a string.
 */
std::string getNextWord(const char *str) {
    // Jump over all blanks
    std::string rtn(str);
    trim(rtn);
    return rtn;
}

double getCurrentTime() {
    long long time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<double>(time) / 1000000.0;
//    return clock() / (double) CLOCKS_PER_SEC;
}


Param param;

Param parseArgs(int nargs, char **args) {
    std::vector<std::string> Algorithms{
        "featpush", "featreuse",
    };

    Param rtn;
    std::unordered_set<std::string> algo_set(Algorithms.begin(), Algorithms.end());

    printf("%s\n", std::string(80, '-').c_str());
    printf("Configs:\n");
    for (unsigned int cnt = 1; cnt < nargs;) {
        char *arg = args[cnt++];
        if (cnt == nargs) {
            printf("Unknown Parameters.\n");
            exit(0);
        }
        unsigned int i = getNextChar(arg);
        if (arg[i] != '-') {
            printf("Unknown Parameters.\n");
            exit(0);
        }
        std::string para = getNextWord(arg + i + 1);
        // printf("-%s\n", para.c_str());
        printf("\t");
        arg = args[cnt++];
        if (para == "algo") {
            rtn.algorithm = std::string(arg);
            std::cout << "Algorithm Parameter: " << rtn.algorithm << std::endl;
            if (algo_set.find(rtn.algorithm) == algo_set.end()) {
                printf("Unknown Algorithm.\n");
                exit(0);
            }
        } else if (para == "data_folder") {
            rtn.data_folder = getNextWord(arg);
            printf("Data Folder: %s\n", rtn.data_folder.c_str());
        } else if (para == "estimation_folder") {
            rtn.output_estimations = true;
            rtn.estimation_folder = getNextWord(arg);
            printf("Estimation Folder: %s\n", rtn.estimation_folder.c_str());
        } else if (para == "graph") {
            rtn.graph_file = getNextWord(arg);
            if (!rtn.data_folder.empty())
                rtn.graph_file = rtn.data_folder + "/" + rtn.graph_file;
            printf("Input Graph File: %s\n", rtn.graph_file.c_str());
        } else if (para == "query") {
            rtn.query_file = getNextWord(arg);
            if (!rtn.data_folder.empty())
                rtn.query_file = rtn.data_folder + "/" + rtn.query_file;
            printf("Input Query File: %s\n", rtn.query_file.c_str());
        } else if (para == "feats") {
            rtn.feature_file = getNextWord(arg);
            if (!rtn.data_folder.empty())
                rtn.feature_file = rtn.data_folder + "/" + rtn.feature_file;
            printf("Feature File: %s\n", rtn.feature_file.c_str());
        } else if (para == "index") {
            auto option = getNextWord(arg);
            if (option == "yes") {
                rtn.index = true;
            } else if (option == "no") {
                rtn.index = false;
            } else {
                printf("Unknown option -%s!\n", option.c_str());
                exit(0);
            }
            std::cout << "With Index: " << rtn.index << "\n";
        } else if (para == "seed") {
            rtn.seed = std::stoi(getNextWord(arg));
            printf("Random Seed: %d\n", rtn.seed);
        } else if (para == "split_num") {
            rtn.split_num = std::stoi(getNextWord(arg));
            printf("Number of Splits: %d\n", rtn.split_num);
        } else if (para == "thread_num") {
            rtn.thread_num = std::stoi(getNextWord(arg));
            printf("Number of threads: %d\n", rtn.thread_num);
        } else if (para == "epsilon") {
            rtn.epsilon = std::stod(getNextWord(arg));
            printf("Epsilon: %.4f\n", rtn.epsilon);
        } else if (para == "alpha") {
            rtn.alpha = std::stod(getNextWord(arg));
            printf("Alpha: %.4f\n", rtn.alpha);
        } else if (para == "gamma") {
            rtn.gamma = std::stod(getNextWord(arg));
            printf("Gamma: %.4f\n", rtn.gamma);
        } else if (para == "base_ratio") {
            rtn.base_ratio = std::stod(getNextWord(arg));
            printf("Base ratio: %.4f\n", rtn.base_ratio);
        }  else {
            printf("Unknown option -%s!\n\n", para.c_str());
            exit(0);
        }
    }
    printf("%s\n", std::string(80, '-').c_str());
    return rtn;
}
