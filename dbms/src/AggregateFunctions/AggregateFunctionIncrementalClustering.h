#pragma once

#include <type_traits>

#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>

#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypesDecimal.h>
#include <Columns/ColumnVector.h>

#include <AggregateFunctions/IAggregateFunction.h>

#include <cmath>
#include <exception>

#include <Columns/ColumnsCommon.h>
#include <Columns/ColumnsNumber.h>
#include <Functions/FunctionHelpers.h>
#include <Common/FieldVisitors.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
    extern const int LOGICAL_ERROR;
}

class ClusteringData
{
public:
    ClusteringData()
    {}

    ClusteringData(UInt32 clusters_num, UInt32 dimensions)
    : clusters_num(clusters_num), dimensions(dimensions), initialized_clusters(0)
    {
        clusters.resize(10 * clusters_num, dimensions);
    }

    void add(const IColumn ** columns, size_t row_num)
    {
        if (initialized_clusters < clusters.size())
        {
            clusters[initialized_clusters++] = Cluster(columns, row_num, dimensions);
            return;
        }

        /// simply searching for the closest cluster over all clusters
        size_t closest_cluster = 0;
        Float64 min_distance = compute_distance_from_point(clusters[0], columns, row_num) * clusters[0].points_num;
        Float64 cur_distance;
        for (size_t i = 1; i != clusters.size(); ++i)
        {
            /// Штрафуем, если в кластере слишком много элементов, чтобы они заполнялись равномерно
            /// Можно попробовать штрафовать на корень? от числа элементов
            cur_distance = compute_distance_from_point(clusters[i], columns, row_num) * clusters[i].points_num;
            if (cur_distance < min_distance)
            {
                min_distance = cur_distance;
                closest_cluster = i;
            }
        }

        clusters[closest_cluster].append_point(columns, row_num, min_distance);
    }
    void add_from_vector(const std::vector<Float64> & coordinates)
    {
        if (initialized_clusters < clusters.size())
        {
            clusters[initialized_clusters++] = Cluster(coordinates);
            return;
        }

        /// simply searching for the closest cluster over all clusters
        size_t closest_cluster = 0;
        Float64 min_distance = compute_distance(clusters[0], coordinates) * clusters[0].points_num;
        Float64 cur_distance;
        for (size_t i = 1; i != clusters.size(); ++i)
        {
            /// Штрафуем, если в кластере слишком много элементов, чтобы они заполнялись равномерно
            /// Можно попробовать штрафовать на корень? от числа элементов
            cur_distance = compute_distance(clusters[i], coordinates) * clusters[i].points_num;
            if (cur_distance < min_distance)
            {
                min_distance = cur_distance;
                closest_cluster = i;
            }
        }

        clusters[closest_cluster].append_point_from_vector(columns, coordinates);
    }

    void merge(const ClusteringData & rhs)
    {
        /// По-простому объединяем попарно элементы (чтобы преположение об одинаковом числе элеметов
        /// в каждом класетер сохранялось (надо бы проверить, насколько это существенно))
        /// Для этого за квадрат ищем ближайших друг к другу соседей

        if (rhs.initialized_clusters < rhs.clusters.size())
        {
            /// Тут мы передаем все точки от rhs к *this
            /// Т.к. rhs константен, то точки будут продублированы еще и там
            for (size_t i = 0; i < rhs.initialized_clusters; ++i)
            {
                add_from_vector(rhs.clusters[i]);
            }
            return;
        }
        if (initialized_clusters < clusters.size())
        {
            std::vector<Cluster> old_clusters = std::move(clusters);

            clusters = rhs.clusters;
            for (size_t i = 0; i < initialized_clusters; ++i)
            {
                add_from_vector(old_clusters[i]);
            }
            initialized_clusters = rhs.initialized_clusters;

            return;
        }

        /// Чтобы все по разу
        std::vector<size_t> rhs_free_indices(rhs.clusters.size());
        for (size_t i = 0; i != rhs.clusters.size(); ++i)
            rhs_free_indices[i] = i;

        size_t idx1;
        size_t idx2;
        for (size_t iter_num = 0; iter_num != clusters.size(); ++iter_num)
        {
            idx1 = 0;
            idx2 = 0;
            Float64 min_dist = compute_distance(clusters[0], rhs.clusters[rhs_free_indices[0]]);
            Float64 cur_dist;
            for (size_t i = 1; i != clusters.size() - iter_num; ++i)
            {
                for (size_t j = 1; j != rhs.clusters.size() - iter_num; ++j)
                {
                    cur_dist = compute_distance(clusters[i], rhs.clusters[rhs_free_indices[j]]);
                    if (cur_dist < min_dist)
                    {
                        min_dist = cur_dist;
                        idx1 = i;
                        idx2 = j;
                    }
                }
            }

            clusters[idx1].merge_cluster(rhs.clusters[rhs_free_indices[idx2]]);
            std::swap(clusters[idx1], clusters[clusters.size() - iter_num - 1]);
            std::swap(rhs_free_indices[idx2], rhs_free_indices[rhs.clusters.size() - iter_num - 1]);
        }
    }

    void write(WriteBuffer & buf) const
    {
        writeBinary(clusters_num, buf);
        writeBinary(dimensions, buf);
        writeBinary(initialized_clusters, buf);
        writeBinary(clusters, buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(clusters_num, buf);
        readinary(dimensions, buf);
        readinary(initialized_clusters, buf);
        readBinary(clusters, buf);
    }

    Float64 predict(const std::vector<Float64> & predict_feature) const
    {
        /// Можно запилить обычную кластеризацию на малом числе элементов

        // TODO
    }

private:
    Float64 compute_distance(std::vector<Float64> & point1, std::vector<Float64> & point2)
    {
        Float64 distance = 0;
        Float64 coordinate_diff;
        for (size_t i = 0; i != point1.size(); ++i)
        {
            coordinate_diff = point1[i] - point2[i];
            distance += coordinate_diff * coordinate_diff;
        }

        return std::sqrt(distance);
    }
    Float64 compute_distance_from_point(std::vector<Float64> & point1, const IColumn ** columns, size_t row_num)
    {
        Float64 distance = 0;
        Float64 coordinate_diff;
        for (size_t i = 0; i < point1.size(); ++i)
        {
            coordinate_diff = point1[i] - static_cast<const ColumnVector<Float64> &>(*columns[i]).getData()[row_num];
            distance += coordinate_diff * coordinate_diff;
        }

        return std::sqrt(distance);
    }

    struct Cluster
    {
        Cluster()
        {}
//        explicit Cluster(UInt32 dimensions)
//        {
//            coordinates.resize(dimensions, 0.0);
//        }
        Cluster(std::vector<Float64> coordinates)
        : coordinates(std::move(coordinates)), points_num(1), total_center_dist(0.0)
        {}

        Cluster(const IColumn ** columns, size_t row_num, size_t dimensions)
        : points_num(1), total_center_dist(0.0)
        {
            coordinates.reserve(dimensions);
            for (size_t i = 0; i != dimensions; ++i)
            {
                coordinates[i] = static_cast<const ColumnVector<Float64> &>(*columns[i]).getData()[row_num];
            }

        }

        void append_point(const IColumn ** columns, size_t row_num, Float64 distance)
        {
            if (points_num == 0)
                throw Exception("This should not be reached 1", ErrorCodes::LOGICAL_ERROR)

            Float64 weight = std::sqrt(Float64{1.0} / points_num);
            for (size_t i = 0; i != coordinates.size(); ++i)
            {
                coordinates[i] = coordinates[i] * weight +
                        static_cast<const ColumnVector<Float64> &>(*columns[i]).getData()[row_num] * (1 - weight);
            }
            total_center_dist += distance;
        }
        void append_point_from_vector(const std::vector<Float64> & new_coordinates)
        {
            if (points_num == 0)
                throw Exception("This should not be reached 2", ErrorCodes::LOGICAL_ERROR)

            ++points_num;
            Float64 weight = std::sqrt(Float64{1.0} / points_num);
            for (size_t i = 0; i != coordinates.size(); ++i)
            {
                coordinates[i] = coordinates[i] * weight + new_coordinates[i] * (1 - weight);
            }
            total_center_dist += distance;
        }
        void merge_cluster(const Cluster & other)
        {
            if (points_num == 0 || other.points_num == 0)
                throw Exception("This should not be reached 3", ErrorCodes::LOGICAL_ERROR)

            Float64 weight = Float64(points_num) / (points_num + other.points_num);
            weight *= weight;

            for (size_t i = 0; i != coordinates.size(); ++i)
            {
                coordinates[i] = coordinates[i] * weight + other.coordinates[i] * (1 - weight);
            }
            points_num += other.points_num;

            /// Поменять
            total_center_dist += other.total_center_dist;
        }

        std::vector<Float64> coordinates;
        UInt32 points_num = 0;
        Float64 total_center_dist = 0.0;
    };

    UInt32 clusters_num;
    UInt32 dimensions;
    Uint32 initialized_clusters;

    std::vector<Cluster> clusters;
};


class AggregateFunctionIncrementalClustering final : public IAggregateFunctionDataHelper<ClusteringData, AggregateFunctionIncrementalClustering>
{
public:
    String getName() const override { return "IncrementalClustering"; }

    explicit AggregateFunctionIncrementalClustering(UInt32 clusters_num, UInt32 dimensions)
    : clusters_num(clusters_num), dimensions(dimensions)
    {}

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeNumber<Float64>>();
    }

    void create(AggregateDataPtr place) const override
    {
        new (place) Data(clusters_num, dimensions);
    }

    void add(AggregateDataPtr place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr place, WriteBuffer & buf) const override
    {
        this->data(place).write(buf);
    }

    void deserialize(AggregateDataPtr place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).read(buf);
    }

    void predictResultInto(ConstAggregateDataPtr place, IColumn & to, Block & block, size_t row_num, const ColumnNumbers & arguments) const
    {
        if (arguments.size() != dimensions)
            throw Exception("Predict got incorrect number of arguments. Got: " + std::to_string(arguments.size()) + ". Required: " + std::to_string(clusters_num),
                            ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        auto &column = dynamic_cast<ColumnVector<Float64> &>(to);

        std::vector<Float64> predict_features(arguments.size());
        for (size_t i = 1; i < arguments.size(); ++i)
        {
            const auto& element = (*block.getByPosition(arguments[i]).column)[row_num];
            if (element.getType() != Field::Types::Float64)
                throw Exception("Prediction arguments must be values of type Float",
                                ErrorCodes::BAD_ARGUMENTS);

            predict_features[i - 1] = element.get<Float64>();
        }
        column.getData().push_back(this->data(place).predict(predict_features));
    }

    void insertResultInto(ConstAggregateDataPtr place, IColumn & to) const override
    {
        std::ignore = place;
        std::ignore = to;
        throw Exception("not implemented", ErrorCodes::LOGICAL_ERROR)
    }

    const char * getHeaderFilePath() const override { return __FILE__; }

private:
    UInt32 clusters_num;
    UInt32 dimensions;
};

}
