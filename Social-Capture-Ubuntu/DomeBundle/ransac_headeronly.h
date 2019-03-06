#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <random>

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif
#include <Eigen/Dense>


class ransac
{
public:

	/** fitting function **/
	typedef void(*TRansacFitFunctor)(
		const Eigen::MatrixXd         &allData,
		const std::vector<size_t>     &useIndices,
		std::vector<Eigen::MatrixXd>  &fitModels);


	/** distance function  **/
	typedef void(*TRansacDistanceFunctor)(
		const Eigen::MatrixXd                 &allData,
		const std::vector< Eigen::MatrixXd >  &testModels,
		const double                          distanceThreshold,
		unsigned int                          &out_bestModelIndex,
		std::vector<bool>                     &status);

	/** degeneracy function **/
	typedef bool(*TRansacDegenerateFunctor)(
		const Eigen::MatrixXd       &allData,
		const std::vector<size_t>   &useIndices);

	/** An implementation of the RANSAC algorithm for robust fitting of models to data.
	*
	*  \param data A DxN matrix with all the observed data. D is the dimensionality of data points and N the number of points.
	*  \param
	*
	*  This implementation is highly inspired on Peter Kovesi's MATLAB scripts (http://www.csse.uwa.edu.au/~pk).
	* \return false if no good solution can be found, true on success.
	*/
	static bool execute(
		const Eigen::MatrixXd	    &data,
		TRansacFitFunctor			fit_func,
		TRansacDistanceFunctor  	dist_func,
		TRansacDegenerateFunctor 	degen_func,
		const double   				distanceThreshold,
		const unsigned int			minimumSizeSamplesToFit,
		std::vector<bool>			&best_status,
		Eigen::MatrixXd             &out_best_model,
		bool						verbose = false,
		const double                prob_good_sample = 0.999,
		const size_t				maxIter = 2000)
	{
		using std::cout;
		using std::vector;
		using std::string;

		const size_t D    = data.rows();  //dimensionality
		const size_t Npts = data.cols();  //the number of the points

#ifdef DEBUG
		assert(minimumSizeSamplesToFit >= 1);
		assert(D >= 1);
		assert(Npts > 1);
#endif

		const size_t maxDataTrials = 100; // Maximum number of attempts to select a non-degenerate data set.


		size_t bestscore  = 0;
		size_t trialcount = 0;
		size_t N          = maxIter;// The number of trials to ensure we pick. Dummy initialisation by maxIter.

		const double eps       = std::numeric_limits<double>::epsilon();
		bool  found_best_model = false;


		vector<bool> status(Npts, false);

		if (best_status.size() != Npts)
			best_status.resize(Npts);
		std::fill(best_status.begin(), best_status.end(), false);


		vector<size_t> ind(minimumSizeSamplesToFit);

		std::random_device rand_dev;
		std::mt19937       rand_engine(rand_dev());



		if (verbose)
			cout << "\n" << std::endl;


		while (trialcount < N)
		{
			// Select at random s datapoints to form a trial model, M.
			// In selecting these points we have to check that they are not in
			// a degenerate configuration.
			bool   degenerate = true;
			size_t count      = 1;
			vector<Eigen::MatrixXd>  models;


			while (degenerate)
			{
				// Generate k random indicies in the range 0..npts-1
				selectKfromN(minimumSizeSamplesToFit, Npts, rand_engine, ind);

				// Test that these points are not a degenerate configuration.
				degenerate = (degen_func != NULL) ? degen_func(data, ind) : false;

				if (!degenerate)
				{
					// Fit model to this random selection of data points.
					// Note that M may represent a set of models that fit the data
					fit_func(data, ind, models);

					// Depending on your problem it might be that the only way you
					// can determine whether a data set is degenerate or not is to
					// try to fit a model and see if it succeeds.  If it fails we
					// reset degenerate to true.
					degenerate = models.empty();
				}

				// Safeguard against being stuck in this loop forever
				if (++count > maxDataTrials)
				{
					if (verbose)
						cout << "[RANSAC] Unable to select a nondegenerate data set\n";
					break;
				}
			}


			if (degenerate)
				continue;


			// Once we are out here we should have some kind of model...
			// Evaluate distances between points and model returning the indices
			// of elements in x that are status.  Additionally, if M is a cell
			// array of possible models 'distfn' will return the model that has
			// the most status.  After this call M will be a non-cell objec
			// representing only one model.
			unsigned int   bestModelIdx = 1000;

			dist_func(data, models, distanceThreshold, bestModelIdx, status);
#ifdef DEBUG
			assert(bestModelIdx < models.size());
#endif



			// Find the number of status to this model.
			const size_t ninliers       = std::count(status.begin(), status.end(), true);
			bool update_estim_num_iters = (trialcount==0); // Always update on the first iteration, regardless of the result (even for ninliers=0)
			if (ninliers > bestscore)
			{
				bestscore   = ninliers;  // Record data for this model
				best_status = status;

				out_best_model   = models[bestModelIdx];
				found_best_model = true;

				update_estim_num_iters = true;
			}


			if (update_estim_num_iters)
			{
				// Update estimate of N, the number of trials to ensure we pick,
				// with probability p, a data set with no outliers.
				double fracinliers = ninliers / static_cast<double>(Npts);
				double pNoOutliers = 1.0 - pow(fracinliers, static_cast<double>(minimumSizeSamplesToFit));

				pNoOutliers = std::max(eps, pNoOutliers);       // Avoid division by -Inf
				pNoOutliers = std::min(1.0 - eps, pNoOutliers); // Avoid division by 0.

				N = static_cast<size_t>(log(1.0 - prob_good_sample) / log(pNoOutliers));
				if (verbose)
					printf("[RANSAC] Iter #%u Estimated number of iters: %u  pNoOutliers = %f  #inliers: %u\n", (unsigned)trialcount, (unsigned)N, pNoOutliers, (unsigned)ninliers);

				N = std::min(N, maxIter);
			}

			++trialcount;

			if (verbose)
				printf("[RANSAC] trial %u out of %u \r", (unsigned int)trialcount, (unsigned int)ceil(static_cast<double>(N)));
		}



		if (verbose)
		{
			if (trialcount == maxIter)
				printf("[RANSAC] Warning: maximum number of trials (%u) reached\n", (unsigned)maxIter);


			if (found_best_model)
				printf("[RANSAC] Finished in %u iterations.\n", (unsigned)trialcount);
			else
				printf("[RANSAC] Warning: Finished without any proper solution.\n");
		}

		return found_best_model;

	}


private:
	static inline void selectKfromN(const size_t K, const size_t N, std::mt19937 &mt, std::vector<size_t> &rand_index)
	{
		if (rand_index.size()!=K)
			rand_index.resize(K);


		//std::uniform_int_distribution<unsigned int> rnd(0, N - 1);
		std::uniform_int_distribution<size_t> rnd(0, N - 1);
		int i = 0;
		while ( i < K )
		{
			size_t n = static_cast<size_t>( rnd(mt) );

			if (std::find(rand_index.begin(), rand_index.end(), n) == rand_index.end())
			{
				rand_index[i] = n;
				i++;
			}
		}

	}


}; // end class