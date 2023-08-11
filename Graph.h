#ifndef GRAPH_H
#define GRAPH_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_set>
#include "BasicDefinition.h"
#include "HelperFunctions.h"
#include <boost/random.hpp>
using namespace std;
const double ALPHA_DEFAULT = 0.2;
class Graph
{

    

public:
	uint n;	//number of nodes
	uint m = 0;	//number of edges

	vector<vector<uint>> inAdj;
	vector<vector<uint>> outAdj;
	uint* indegree;
	uint* outdegree;
  	vector<uint>indices;
  	vector<uint>indptr;

    double alpha = 0.2;
    VertexIdType numOfVertices = 0;
    EdgeSizeType numOfEdges = 0;
    VertexIdType num_deadend_vertices = 0;
    VertexIdType sid = 0;
    VertexIdType dummy_id = 0;
    EdgeSizeType max_size_edge_list = 0;
    std::vector<VertexIdType> out_degrees;
    std::vector<VertexIdType> in_degrees;
    std::vector<VertexIdType> start_pos_in_out_neighbor_lists;
    std::vector<VertexIdType> start_pos_in_appearance_pos_lists;
    std::vector<VertexIdType> out_neighbors_lists;
    std::vector<VertexIdType> appearance_pos_lists;
    std::vector<VertexIdType> deadend_vertices;
	// Graph()
	// {
	// }
	// ~Graph()
	// {
	// }
    Graph() = default;

    ~Graph() = default;
	void insertEdge(uint from, uint to) {
		outAdj[from].push_back(to);
		inAdj[to].push_back(from);
		outdegree[from]++;
		indegree[to]++;
	}

	void deleteEdge(uint from, uint to) {
		uint j;
		for (j=0; j < indegree[to]; j++) {
			if (inAdj[to][j] == from) {
				break;
			}
		}
		inAdj[to].erase(inAdj[to].begin()+j);
		indegree[to]--;

		for (j=0; j < outdegree[from]; j++) {
			if (outAdj[from][j] == to) {
				break;
			}
		}

		outAdj[from].erase(outAdj[from].begin() + j);
		outdegree[from]--;
	}

	int isEdgeExist(uint u, uint v) {
		for (uint j = 0; j < outdegree[u]; j++) {
			if (outAdj[u][j] == v) {
				return -1;
			}
		}
		return 1;
	}

	void inputGraph(string path, string dataset, uint nodenum, uint edgenum)
	{
    n = nodenum;
    m = edgenum;
    indices=vector<uint>(m);
    indptr=vector<uint>(n+1);
    //string dataset_el="data/"+dataset+"_adj_el.txt";
    string dataset_el=path+dataset+"_adj_el.txt";
    const char *p1=dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb"))
    {
        size_t rtn = fread(indices.data(), sizeof indices[0], indices.size(), f1);
        if(rtn!=m)
            cout<<"Error! "<<dataset_el<<" Incorrect read!"<<endl;
            cout<<" rtn: "<<rtn<<" m:"<<m<<endl;
        fclose(f1);
    }
    else
    {
        cout<<dataset_el<<" Not Exists."<<endl;
        exit(1);
    }
    string dataset_pl=path+dataset+"_adj_pl.txt";
    const char *p2=dataset_pl.c_str();

    if (FILE *f2 = fopen(p2, "rb"))
    {
        size_t rtn = fread(indptr.data(), sizeof indptr[0], indptr.size(), f2);
        if(rtn!=n+1)
            cout<<"Error! "<<dataset_pl<<" Incorrect read!"<<endl;
        fclose(f2);
    }
    else
    {
        cout<<dataset_pl<<" Not Exists."<<endl;
        exit(1);
    }
		indegree=new uint[n];
		outdegree=new uint[n];
        clock_t t1=clock();
		for(uint i=0;i<n;i++)
		{
			indegree[i] = indptr[i+1]-indptr[i];
            // MSG(indegree[i]);
            outdegree[i] = indptr[i+1]-indptr[i];
            vector<uint> templst(indices.begin() + indptr[i],indices.begin() + indptr[i+1]);//https://blog.csdn.net/bymaymay/article/details/80842416
            outAdj.push_back(templst);
            inAdj.push_back(templst);
		}
		
		clock_t t2=clock();
		cout<<"m="<<m<<endl;
		cout<<"reading in graph takes "<<(t2-t1)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
	}
    inline static unsigned long lrand() {
        static boost::taus88 rngG(time(0));
        return rngG();
        //return rand();
        // return sfmt_genrand_uint32(&sfmtSeed);
    }
    inline static bool drand(){
        static boost::bernoulli_distribution <> bernoulli(ALPHA_DEFAULT);
        static boost::lagged_fibonacci607 rngG(time(0));
        static boost::variate_generator<boost::lagged_fibonacci607&, boost::bernoulli_distribution<> > bernoulliRngG(rngG, bernoulli);

        return bernoulliRngG();
        //return rand()*1.0f/RAND_MAX;
        // return sfmt_genrand_real1(&sfmtSeed);
    }
    inline int random_walk(int start){
        int cur = start;
        unsigned long k;
        if(getInSize(start)==0){
            return start;
        }
        while (true) {
            if (drand()) {
                return cur;
            }
            if (getInSize(cur)>0){
                k = lrand()%getInSize(cur)>0;
                cur = getInVert(cur, k);
            }
            else{
                cur = start;
            }
        }
    }
	void inputGraph_fromedgelist(string path, string dataset, uint nodenum ,uint edgenum)
	{
		
        n = (unsigned long) nodenum;
        m = (unsigned long) edgenum;
        string filename=path+dataset+"_adj.txt";
        indices=vector<uint>(m);
        const char *p1=filename.c_str();
        if (FILE *f1 = fopen(p1, "rb"))
        {
            size_t rtn = fread(indices.data(), sizeof indices[0], indices.size(), f1);
            if(rtn!=m)
                cout<<"Error! "<<filename<<" Incorrect read!"<<endl;
            fclose(f1);
        }
        else
        {
            cout<<filename<<" Not Exists."<<endl;
            exit(1);
        }
        // string filename = "data/"+dataset+"_adj.txt";
		ifstream infile(filename.c_str());

		indegree=new uint[n];
		outdegree=new uint[n];
		for(uint i=0;i<n;i++)
		{
			indegree[i]=0;
			outdegree[i]=0;
		}
		//read graph and get degree info
		uint from;
		uint to;
		while(infile>>from>>to)
		{
			outdegree[from]++;
			indegree[to]++;
		}

		cout<<"..."<<endl;

		for (uint i = 0; i < n; i++)
		{
			vector<uint> templst;
			inAdj.push_back(templst);
			outAdj.push_back(templst);
		}

		infile.clear();
		infile.seekg(0);

		clock_t t1=clock();

		while(infile>>from>>to)
		{
			outAdj[from].push_back(to);
			inAdj[to].push_back(from);
		}
		infile.close();
		clock_t t2=clock();
        cout<<"n="<<n<<endl;
		cout<<"m="<<m<<endl;
		cout<<"reading in graph takes "<<(t2-t1)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
	} 

	uint getInSize(uint vert){
		return indegree[vert];
	}
	uint getInVert(uint vert, uint pos){
		return inAdj[vert][pos];
	}
	uint getOutSize(uint vert){
		return outdegree[vert];
	}
	uint getOutVert(uint vert, uint pos){
		return outAdj[vert][pos];
	}
  	vector<uint> getOutAdjs(uint vert){
        return outAdj[vert];
    }

// Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank

	inline size_t get_num_dead_end() const {
        return deadend_vertices.size();
    }

    inline void set_dummy_out_degree_zero() {
        out_degrees[dummy_id] = 0;
        start_pos_in_out_neighbor_lists[dummy_id + 1] = start_pos_in_out_neighbor_lists[dummy_id];
    }

    inline void set_dummy_neighbor(const VertexIdType &_id) {
        out_degrees[dummy_id] = 1;
        start_pos_in_out_neighbor_lists[dummy_id + 1] = start_pos_in_out_neighbor_lists[dummy_id] + 1;
        out_neighbors_lists[start_pos_in_out_neighbor_lists[dummy_id]] = _id;
    }

    inline void reset_set_dummy_neighbor() {
        out_degrees[dummy_id] = 0;
        out_neighbors_lists[start_pos_in_out_neighbor_lists[dummy_id]] = dummy_id;
        set_dummy_out_degree_zero();
    }

    inline const VertexIdType &get_dummy_id() const {
        return dummy_id;
    }

    inline const VertexIdType &get_sid() const {
        return sid;
    }

    inline const PageRankScoreType &get_alpha() const {
        return alpha;
    }

    inline void set_alpha(const PageRankScoreType _alpha = 0.2) {
        alpha = _alpha;
    }

    inline void fill_dead_end_neighbor_with_id(const VertexIdType &_id) {
        for (VertexIdType index = 0; index < num_deadend_vertices; ++index) {
            const VertexIdType &id = deadend_vertices[index];
            const VertexIdType &start = start_pos_in_out_neighbor_lists[id];
            out_neighbors_lists[start] = _id;
        }
    }

    inline void fill_dead_end_neighbor_with_id() {
        //std::cout<< "num_deadend_vertices: " << num_deadend_vertices <<std::endl;
        for (VertexIdType index = 0; index < num_deadend_vertices; ++index) {
            const VertexIdType &id = deadend_vertices[index];
            const VertexIdType &start = start_pos_in_out_neighbor_lists[id];
            VertexIdType _id = rand()%numOfVertices;
            out_neighbors_lists[start] = _id;
        }
    }


    inline void change_in_neighbors_adj(const VertexIdType &_sid, const VertexIdType &_target) {
        const VertexIdType &idx_start = start_pos_in_appearance_pos_lists[_sid];
        const VertexIdType &idx_end = start_pos_in_appearance_pos_lists[_sid + 1];
        for (VertexIdType index = idx_start; index < idx_end; ++index) {
            out_neighbors_lists[appearance_pos_lists[index]] = _target;
        }
    }

    inline void restore_neighbors_adj(const VertexIdType &_sid) {
        const VertexIdType &idx_start = start_pos_in_appearance_pos_lists[_sid];
        const VertexIdType &idx_end = start_pos_in_appearance_pos_lists[_sid + 1];
        for (VertexIdType index = idx_start; index < idx_end; ++index) {
            out_neighbors_lists[appearance_pos_lists[index]] = _sid;
        }
    }

    inline void set_source_and_alpha(const VertexIdType _sid, const PageRankScoreType _alpha) {
        sid = _sid;
        alpha = _alpha;
//        fill_dead_end_neighbor_with_id(_sid);
    }


    inline const VertexIdType &getNumOfVertices() const {
        return numOfVertices;
    }

    /**
     * @param _vid
     * @return return the original out degree
     */
    inline const VertexIdType &original_out_degree(const VertexIdType &_vid) const {
        assert(_vid < numOfVertices);
        return out_degrees[_vid];
    }

    inline const VertexIdType &get_neighbor_list_start_pos(const VertexIdType &_vid) const {
        assert(_vid < numOfVertices + 2);
        return start_pos_in_out_neighbor_lists[_vid];
    }


    inline const VertexIdType &getOutNeighbor(const VertexIdType &_index) const {
//        if (_index >= start_pos_in_out_neighbor_lists[dummy_id + 1]) {
//            MSG("Time to check " __FILE__)
//            MSG(__LINE__)
//        }
        assert(_index < start_pos_in_out_neighbor_lists[dummy_id + 1]);
        return out_neighbors_lists[_index];
    }

    inline const EdgeSizeType &getNumOfEdges() const {
        return numOfEdges;
    }


    void read_for_speedppr(string path, string dataset, uint nodenum ,uint edgenum) {
        // {
        //     std::string line;
        //     std::ifstream attribute_file(_attribute_file);
        //     if (attribute_file.is_open()) {
        //         std::getline(attribute_file, line);
        //         size_t start1 = line.find_first_of('=');
        //         numOfVertices = std::stoul(line.substr(start1 + 1));
        //         std::getline(attribute_file, line);
        //         size_t start2 = line.find_first_of('=');
        //         numOfEdges = std::stoul(line.substr(start2 + 1));
        //         dummy_id = numOfVertices;
        //         // printf("The Number of Vertices: %" IDFMT "\n", numOfVertices);
        //         // printf("The Number of Edges: %" IDFMT "\n", numOfEdges);
        //         attribute_file.close();
        //     } else {
        //         printf(__FILE__ "; LINE %d; File Not Exists.\n", __LINE__);
        //         std::cout << _attribute_file << std::endl;
        //         exit(1);
        //     }
        // }

        numOfVertices = (VertexIdType) nodenum;
        numOfEdges = (VertexIdType) edgenum;
        dummy_id = numOfVertices;
        string _graph_file=path+dataset+"_adj.txt";
        clock_t start=clock();
        // create temporary graph
        ifstream infile(_graph_file.c_str());
        VertexIdType num_lines = 0;
        std::vector<Edge> edges;
        if (std::FILE *f = std::fopen(_graph_file.c_str(), "rb")) {
            // size_t rtn = std::fread(edges.data(), sizeof edges[0], edges.size(), f);

            for (VertexIdType fromId, toID; infile >> fromId >> toID;) {
                edges.emplace_back(fromId, toID);
                if (++num_lines % 5000000 == 0) { printf("%zu Valid Lines Read.\n", num_lines); }
            }
            if(edges.size()!=2*numOfEdges)
                cout<<"Error! "<<_graph_file<<" Incorrect read!"<<endl;
            // printf("edges.size(): %zu\n %zu\n", edges[0].from_id, edges[0].to_id);
            printf("Returned Value of fread: %zu\n", edges.size());
            std::fclose(f);
        } else {
            printf("Graph::read; File Not Exists.\n");
            std::cout << _graph_file << std::endl;
            exit(1);
        }
        clock_t end=clock();
        printf("Time Used For Loading BINARY : %.2f\n", (end-start)/(1.0*CLOCKS_PER_SEC));

        // read the edges
        // the ids must be in the range from [0 .... the number of vertices - 1];
        numOfEdges = 0;
        out_degrees.clear();
        out_degrees.resize(numOfVertices + 2, 0);
        in_degrees.clear();
        in_degrees.resize(numOfVertices + 2, 0);
        for (auto &edge : edges) {
            const VertexIdType &from_id = edge.from_id;
            const VertexIdType &to_id = edge.to_id;
            // remove self loop
            if (from_id != to_id) {
                //the edge read is a directed one
                ++out_degrees[from_id];
                ++in_degrees[to_id];
                ++numOfEdges;
            }
        }
        /* final count */
       printf("%d-th Directed Edge Processed.\n", numOfEdges);

        // sort the adj list
//        for (auto &neighbors : matrix) {
//            std::sort(neighbors.begin(), neighbors.end());
//        }

        // process the dead_end
        VertexIdType degree_max = 0;
        deadend_vertices.clear();
        for (VertexIdType i = 0; i < numOfVertices; ++i) {
            if (out_degrees[i] == 0) {
                deadend_vertices.emplace_back(i);
            }
            degree_max = std::max(degree_max, out_degrees[i]);
        }
        num_deadend_vertices = deadend_vertices.size();
        printf("The number of dead end vertices:%" IDFMT "\n", num_deadend_vertices);

        // process pos_list list
        start_pos_in_appearance_pos_lists.clear();
        start_pos_in_appearance_pos_lists.resize(numOfVertices + 2, 0);
        for (VertexIdType i = 0, j = 1; j < numOfVertices; ++i, ++j) {
            start_pos_in_appearance_pos_lists[j] = start_pos_in_appearance_pos_lists[i] + in_degrees[i];
        }
        start_pos_in_appearance_pos_lists[numOfVertices] = numOfEdges;

        // process out list
        start_pos_in_out_neighbor_lists.clear();
        start_pos_in_out_neighbor_lists.resize(numOfVertices + 2, 0);
        for (VertexIdType current_id = 0, next_id = 1; next_id < numOfVertices + 1; ++current_id, ++next_id) {
            start_pos_in_out_neighbor_lists[next_id] =
                    start_pos_in_out_neighbor_lists[current_id] + std::max(out_degrees[current_id], (VertexIdType) 1u);
        }
        // process dummy vertex
        assert(start_pos_in_out_neighbor_lists[numOfVertices] == numOfEdges + deadend_vertices.size());
        out_degrees[dummy_id] = 0;
        start_pos_in_out_neighbor_lists[numOfVertices + 1] = start_pos_in_out_neighbor_lists[numOfVertices];

        // compute the positions
        std::vector<VertexIdType> out_positions_to_fill(start_pos_in_out_neighbor_lists.begin(),
                                                        start_pos_in_out_neighbor_lists.end());
        // fill the edge list
        out_neighbors_lists.clear();
        out_neighbors_lists.resize(numOfEdges + num_deadend_vertices + degree_max, 0);
        VertexIdType edges_processed = 0;
        VertexIdType msg_gap = std::max((VertexIdType) 1u, numOfEdges / 10);
        std::vector<std::pair<VertexIdType, VertexIdType>> position_pair;
        position_pair.reserve(numOfEdges);
        for (auto &edge : edges) {
            const VertexIdType &from_id = edge.from_id;
            const VertexIdType &to_id = edge.to_id;
            // remove self loop
            if (from_id != to_id) {
                VertexIdType &out_position = out_positions_to_fill[from_id];
                assert(out_position < out_positions_to_fill[from_id + 1]);
                out_neighbors_lists[out_position] = to_id;
                position_pair.emplace_back(to_id, out_position);
                ++out_position;
                ++edges_processed;
                // if (edges_processed % msg_gap == 0) {
                //     printf("%u edges processed.\n", edges_processed);
                // }
            }
        }
        edges.clear();
        MSG(edges_processed);

        // use reverse position
        std::vector<VertexIdType> in_positions_to_fill(start_pos_in_appearance_pos_lists.begin(),
                                                       start_pos_in_appearance_pos_lists.end());
        in_positions_to_fill[numOfVertices] = numOfEdges;
        // const double time_sort_start = getCurrentTime();
        std::sort(position_pair.begin(), position_pair.end(), std::less<>());
        // const double time_sort_end = getCurrentTime();
//        MSG(time_sort_end - time_sort_start);
        appearance_pos_lists.clear();
        appearance_pos_lists.resize(numOfEdges + num_deadend_vertices + degree_max, 0);
        VertexIdType in_pos_pair = 0;
        for (const auto &pair : position_pair) {
            const VertexIdType &to_id = pair.first;
            const VertexIdType &pos = pair.second;
            VertexIdType &in_position = in_positions_to_fill[to_id];
            assert(in_position < in_positions_to_fill[to_id + 1]);
            appearance_pos_lists[in_position] = pos;
            ++in_position;
            // if (++in_pos_pair % msg_gap == 0) {
            //     MSG(in_pos_pair);
            // }
        }

        // fill the dummy ids
        for (const VertexIdType &id : deadend_vertices) {
            out_neighbors_lists[out_positions_to_fill[id]++] = dummy_id;
        }
        assert(get_neighbor_list_start_pos(get_dummy_id()) ==
               get_neighbor_list_start_pos(get_dummy_id() + 1));
        // const double time_end = getCurrentTime();
        // printf("Graph Build Finished. TIME: %.4f\n", time_end - start);
        printf("%s\n", std::string(80, '-').c_str());
    }

    void show() const {
        // we need to show the dummy
        const VertexIdType num_to_show = std::min(numOfVertices + 1, (VertexIdType) 50u);
        // show the first elements
        show_vector("The Out Degrees of The Vertices:",
                    std::vector<VertexIdType>(out_degrees.data(), out_degrees.data() + num_to_show));
        show_vector("The Start Positions of The Vertices in Out Neighbor Lists:",
                    std::vector<VertexIdType>(start_pos_in_out_neighbor_lists.data(),
                                              start_pos_in_out_neighbor_lists.data() + num_to_show));
        show_vector("The In Degrees of The Vertices:",
                    std::vector<VertexIdType>(in_degrees.data(), in_degrees.data() + num_to_show));
        show_vector("The Start Positions of The Vertices in Appearance List:",
                    std::vector<VertexIdType>(start_pos_in_appearance_pos_lists.data(),
                                              start_pos_in_appearance_pos_lists.data() + num_to_show));
        // assume that the number of vertices >= the number of edges; otherwise, there is a potential bug here.
        show_vector("Out Neighbor Lists:",
                    std::vector<VertexIdType>(out_neighbors_lists.data(),
                                              out_neighbors_lists.data() +
                                              std::min(numOfEdges + num_deadend_vertices, (VertexIdType) 50u)));
        show_vector("The Appearance Positions of Vertices in the Out Neighbor Lists:",
                    std::vector<VertexIdType>(appearance_pos_lists.data(),
                                              appearance_pos_lists.data() + std::min(numOfEdges, (VertexIdType) 50u)));
//        show_vector("The adj list of the middel vertex", matrix[numOfVertices / 2]);
        printf("The position the id appears in outNeighbor List:\n");
        for (VertexIdType id = 0; id < numOfVertices; ++id) {
            const VertexIdType &idx_start = start_pos_in_appearance_pos_lists[id];
            const VertexIdType &idx_end = start_pos_in_appearance_pos_lists[id + 1];
            printf("Id:%" IDFMT ";\tPositions: ", id);
            for (VertexIdType index = idx_start; index < idx_end; ++index) {
                printf("%" IDFMT ", ", appearance_pos_lists[index]);
            }
            printf("\n");
        }
        show_vector("Dead End Vertices List:",
                    std::vector<VertexIdType>(deadend_vertices.data(),
                                              deadend_vertices.data() +
                                              std::min(num_deadend_vertices, (VertexIdType) 50u)));
        printf("\n%s\n", std::string(80, '-').c_str());
    }
};
class CleanGraph {
    VertexIdType numOfVertices = 0;
    EdgeSizeType numOfEdges = 0;
public:

    void clean_graph(std::string &_input_file,
                     std::string &_data_folder) {
        _input_file=_data_folder+_input_file+"_adj.txt";
        std::ifstream inf(_input_file.c_str());
        if (!inf.is_open()) {
            printf("CleanGraph::clean_graph; File not exists.\n");
            printf("%s\n", _input_file.c_str());
            exit(1);
        }
        // status indicator
        // printf("\nReading Input Graph\n");

        std::string line;
        /**
         * skip the headers, we assume the headers are the comments that
         * begins with '#'
         */
        while (std::getline(inf, line) && line[0] == '#') {}
        if (line.empty() || !isdigit(line[0])) {
            printf("Error in CleanGraph::clean_graph. Raw File Format Error.\n");
            printf("%s\n", line.c_str());
            exit(1);
        }
        // create temporary graph
        std::vector<Edge> edges;
        numOfEdges = 0;
        /**
         * read the raw file
         */
        size_t num_lines = 0;
        // process the first line
        {
            VertexIdType fromId, toID;
            ++num_lines;
            size_t end = 0;
            fromId = std::stoul(line, &end);
            toID = std::stoul(line.substr(end));
            // remove self-loops
            edges.emplace_back(fromId, toID);
        }
        // read the edges
        for (VertexIdType fromId, toID; inf >> fromId >> toID;) {
            edges.emplace_back(fromId, toID);
            if (++num_lines % 5000000 == 0) { printf("%zu Valid Lines Read.\n", num_lines); }
        }

        // close the file
        inf.close();
        /* final count */
        printf("%zu Lines Read.\n", num_lines);
        numOfEdges = edges.size();
        printf("%" IDFMT "-th Non-Self Loop Edges.\n", numOfEdges);
        printf("Finish Reading.\n");
        printf("%s\n", std::string(80, '-').c_str());

        // find the maximum id
        size_t id_max = 0;
        size_t id_min = std::numeric_limits<uint64_t>::max();
        for (const auto &pair : edges) {
            id_max = std::max(id_max, (size_t) std::max(pair.from_id, pair.to_id));
            id_min = std::min(id_min, (size_t) std::min(pair.from_id, pair.to_id));
        }
        printf("Minimum ID: %zu, Maximum ID: %zu\n", id_min, id_max);
        if (id_max >= std::numeric_limits<uint64_t>::max()) {
            printf("Warning: Change VertexIdType First.\n");
            exit(1);
        }
        const VertexIdType one_plus_id_max = id_max + 1;
        std::vector<VertexIdType> out_degree(one_plus_id_max, 0);
        std::vector<VertexIdType> in_degree(one_plus_id_max, 0);
        // compute the degrees.
        for (const auto &edge : edges) {
            ++out_degree[edge.from_id];
            ++in_degree[edge.to_id];
        }
        // count the number of dead-end vertices
        VertexIdType original_dead_end_num = 0;
        VertexIdType num_isolated_points = 0;
        VertexIdType max_degree = 0;
        for (VertexIdType id = 0; id < one_plus_id_max; ++id) {
            if (out_degree[id] == 0) {
                ++original_dead_end_num;
                if (in_degree[id] == 0) {
                    ++num_isolated_points;
                }
            }
            // compute maximum out degree
            max_degree = std::max(out_degree[id], max_degree);
        }
        printf("The number of dead end vertices: %" IDFMT "\n", original_dead_end_num);
        printf("The number of isolated points: %" IDFMT "\n", num_isolated_points);
        printf("The maximum out degree is: %" IDFMT "\n", max_degree);

        // we assume the vertice ids are in the arrange of 0 ... numOfVertices - 1
        numOfVertices = one_plus_id_max;

        // sort the edges
        std::sort(edges.begin(), edges.end());

        // Write the attribute file
        numOfEdges = edges.size();
        std::string attribute_file = _data_folder + '/' + "attribute.txt";
        if (std::FILE *file = std::fopen(attribute_file.c_str(), "w")) {
            std::fprintf(file, "n=%" IDFMT "\nm=%" IDFMT "\n", numOfVertices, numOfEdges);
            std::fclose(file);
        } else {
            printf("Graph::clean_graph; File Not Exists.\n");
            printf("%s\n", attribute_file.c_str());
            exit(1);
        }

        // write the graph in binary
        std::string graph_bin_file = _data_folder + '/' + "graph.bin";
        if (std::FILE *file = std::fopen(graph_bin_file.c_str(), "wb")) {
            std::fwrite(edges.data(), sizeof edges[0], edges.size(), file);
            printf("Writing Binary Finished.\n");
            std::fclose(file);
        } else {
            printf("Graph::clean_graph; File Not Exists.\n");
            printf("%s\n", graph_bin_file.c_str());
            exit(1);
        }
        printf("%s\n", std::string(80, '-').c_str());
    }
};

class Config {
public:
    string graph_alias;
    string graph_location;

    string action = ""; // query/generate index, etc..

    string prefix = "d:\\dropbox\\research\\data\\";

    string version = "vector";

    bool multithread = false;
    bool with_rw_idx = false;
	bool with_baton = false;
	bool exact = false;
	bool reuse = false;
	bool power_iteration=false;
	bool adaptive=false;
	bool alter_idx=false;
    bool opt = false;
    bool remap = false;
    bool force_rebuild = false;
    bool balanced = false;
    bool with_fora = false;
    bool no_rebuild = false;
    // int num_rw = 10;

    double omega; // 1/omega  omega = # of random walk
    double rmax; // identical to r_max


    double lambda_q = 30; //ratio of query
    double lambda_u = 200; //ratio of update
    double rate = 1.0; //ratio of query/update
    double simulation_time = 10.0; //simulation time
    int runs = 1;//multiple runs
    int linear_runs = 5;
    double test_beta1 = 1.0;
    double beta1 = 1.0, beta2 = 1.0; //the optimization parameters
    bool test_throughput = false; //true when test the throuput, otherwise test the response time
    double response_t = 0.5;

    unsigned int query_size = 200;
	unsigned int update_size = 200;
	unsigned int check_size = 0;
    unsigned int check_from = 500;


    unsigned int max_iter_num = 100;
    bool reverse_flag = false;
    double pfail = 0;
    double dbar = 0;
    double epsilon = 0.5;
    double delta = 0;
	double beta = 1.0;
	double sigma = 0.5;
    double errorlimiter = 1.0;
    double insert_ratio = 1.0;
	double rbmax_p = 1e-5;
    double rbmax_n = -1e-5;
    double rbmax;
    double theta;
	double n = 2.0;
    int nodes, edges;
    int show_each;
    
    long graph_n = 0;
    pair<double, double> mv_query, mv_update;
    unsigned int k = 500;
    double ppr_decay_alpha = 0.77;

    double rw_cost_ratio = 8.0;//8.0;

    double rmax_scale = 1.0;
    double multithread_param = 1.0;

    string algo;

    double alpha = ALPHA_DEFAULT;

    string exact_pprs_folder;

    unsigned int hub_space_consum = 1;
};

#endif