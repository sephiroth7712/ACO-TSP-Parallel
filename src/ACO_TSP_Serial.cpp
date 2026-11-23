#include <bits/stdc++.h>
using namespace std;
struct city
{
    float x;
    float y;
};

int N, start, iterations, n_ants;
struct city *cities;
int *tour, **ant_routes;
float **pheromones, **d_pheromones;
float min_tour_cost = FLT_MAX, tau_max, tau_min, ro, alpha, bet, q;
bool ranSolver = false;

float get_distance(int source_city, int dest_city)
{
    struct city a = cities[source_city];
    struct city b = cities[dest_city];
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void init()
{
    tour = new int[N];
    ant_routes = new int *[n_ants];
    pheromones = new float *[N];
    d_pheromones = new float *[N];

    for (int i = 0; i < N; i++)
    {
        pheromones[i] = new float[N];
        d_pheromones[i] = new float[N];
        for (int j = 0; j < N; j++)
        {
            pheromones[i][j] = tau_max;
            d_pheromones[i][j] = 0.0;
        }
    }

    for (int i = 0; i < n_ants; i++)
    {
        ant_routes[i] = new int[N];
        for (int j = 0; j < N; j++)
        {
            ant_routes[i][j] = -1;
        }
    }
}

bool is_visited(int ant_id, int city_id)
{
    for (int i = 0; i < N; i++)
    {
        if (ant_routes[ant_id][i] == -1)
        {
            break;
        }
        else if (ant_routes[ant_id][i] == city_id)
        {
            return true;
        }
    }

    return false;
}

float get_probability(int source_city, int dest_city)
{
    float eta = pow(1 / get_distance(source_city, dest_city), bet);
    float tau = pow(pheromones[source_city][dest_city], alpha);
    return eta * tau;
}

int get_next_city(int ant_id, int current_city)
{
    int next_city = -1;
    double *probabilties = new double[N];
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        bool visited = is_visited(ant_id, i);
        if (i == current_city || visited)
            probabilties[i] = 0;
        else if (!visited)
            probabilties[i] = get_probability(current_city, i);

        sum += probabilties[i];
    }
    double choice = (double)rand() / (double)(RAND_MAX);
    double cum_sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        if (std::isinf(probabilties[i]))
        {
            next_city = i;
            break;
        }
        cum_sum += probabilties[i] / sum;

        if (choice <= cum_sum)
        {
            next_city = i;
            break;
        }
    }
    free(probabilties);
    return next_city;
}

void ant_journey(int ant_id)
{
    int current_city = 0;
    int index = 0;
    ant_routes[ant_id][index] = current_city;
    while (true)
    {
        int next_city = get_next_city(ant_id, current_city);
        if (next_city == -1 || index >= N)
        {
            break;
        }
        ant_routes[ant_id][++index] = next_city;
        current_city = next_city;
    }
}

float get_ant_journey_length(int ant_id)
{
    float distance = 0.0;
    for (int i = 0; i < N - 1; i++)
    {
        distance += get_distance(ant_routes[ant_id][i], ant_routes[ant_id][i + 1]);
    }
    distance += get_distance(ant_routes[ant_id][N - 1], ant_routes[ant_id][0]);
    return distance;
}

void update_pheromones()
{
    float min_distance = FLT_MAX;
    int min_ant_id = -1;
    for (int i = 0; i < n_ants; i++)
    {
        float distance = get_ant_journey_length(i);
        if (distance < min_distance)
        {
            min_distance = distance;
            min_ant_id = i;
        }
    }

    for (int i = 0; i < N - 1; i++)
    {
        int source_city_id = ant_routes[min_ant_id][i];
        int dest_city_id = ant_routes[min_ant_id][i + 1];
        d_pheromones[source_city_id][dest_city_id] = q / min_distance;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            pheromones[i][j] = max(min((1 - ro) * pheromones[i][j] + d_pheromones[i][j], tau_max), tau_min);
            d_pheromones[i][j] = 0.0;
            // cout << pheromones[i][j] << " ";
        }
        // cout << endl;
    }
}

void optimize()
{
    float iter_min_dist = FLT_MAX;
    int *iter_best_tour = new int[N];
    for (int i = 0; i < iterations; i++)
    {
        // cout << "Running iteration: " << i << endl;
        for (int j = 0; j < n_ants; j++)
        {
            // cout << "Processing ant: " << j << endl;
            ant_journey(j);

            float distance = get_ant_journey_length(j);

            if (distance < min_tour_cost)
            {
                min_tour_cost = distance;
                for (int k = 0; k < N; k++)
                {
                    tour[k] = ant_routes[j][k];
                }
            }

            if (distance < iter_min_dist)
            {
                iter_min_dist = distance;
                for (int k = 0; k < N; k++)
                {
                    iter_best_tour[k] = ant_routes[j][k];
                }
            }
            // cout << "Ant: " << j << " has ended!" << endl;
        }
        if (i % 500 == 0)
        {
            cout << "Best Route in iteration: " << i << " length: " << iter_min_dist << endl;
        }
        // cout << "Iteration Best" << endl;
        // cout << "Route Length: " << iter_min_dist << endl;
        // for (int k = 0; k < N; k++)
        // {
        //     cout << iter_best_tour[k] << " ";
        // }
        // cout << endl;
        // cout << "updating PHEROMONES . . ." << endl;
        update_pheromones();
        // cout << " done!" << endl;

        for (int j = 0; j < n_ants; j++)
        {
            for (int k = 0; k < N; k++)
            {
                ant_routes[j][k] = -1;
            }
        }
        iter_min_dist = FLT_MAX;
    }
    free(iter_best_tour);
    free(pheromones);
    free(d_pheromones);
}

void read_file(string file_name)
{
    ifstream file(file_name);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << file_name << endl;
    }
    string line;
    bool is_coords = false;
    while (getline(file, line))
    {
        if (line == "EOF" || line == "EOF\n")
        {
            break;
        }
        istringstream iss(line);
        string keyword;
        iss >> keyword;
        if (keyword == "DIMENSION")
        {
            string temp;
            iss >> temp;
            iss >> N;
            cities = new struct city[N];
        }
        else if (keyword == "NODE_COORD_SECTION")
        {
            is_coords = true;
        }
        else if (is_coords)
        {
            int id = stoi(keyword);
            float x, y;
            iss >> x >> y;
            cities[id - 1].x = x;
            cities[id - 1].y = y;
        }
    }
    file.close();
}

int main(int argc, char *argv[])
{

    if (argc != 10)
    {
        cout << "Usage: ./program datasets/filename.tsp n_ants n_iterations alpha beta ro tau_max tau_min q" << endl;
        exit(0);
    }

    string file_name = argv[1];
    read_file(file_name);

    n_ants = strtol(argv[2], NULL, 10);
    iterations = strtol(argv[3], NULL, 10);
    alpha = strtof(argv[4], NULL);
    bet = strtof(argv[5], NULL);
    ro = strtof(argv[6], NULL);
    tau_max = strtof(argv[7], NULL);
    tau_min = strtof(argv[8], NULL);
    q = strtof(argv[9], NULL);

    srand(time(0));
    struct timespec start_time, end_time;

    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

    init();
    start = 0;

    optimize();

    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);

    uint64_t time_difference = (1000000000L * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_nsec - start_time.tv_nsec) / 1e6;
    printf("Solving the TSP with %d cities required %llu ms CPU time for single-thread\n", N, (long long unsigned int)time_difference);

    printf("Cities:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%f,%f\n", cities[i].x, cities[i].y);
    }
    printf("Best tour cost = %f\n", min_tour_cost);
    printf("Order of cities: \n");
    for (int i = 0; i < N; i++)
    {
        cout << tour[i] << ",";
    }
    cout << tour[0];
    cout << endl;

    free(tour);
    free(cities);
}