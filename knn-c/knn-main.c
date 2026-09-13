#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

typedef struct
{
    float x_cord;
    float y_cord;
    int type;
} Point;

typedef struct
{
    float dist;
    int type;
    int count;
    bool firstInstance;
} Distance;


int predict_knn(Point *dataset, int n, Point new_point, int k, int p)
{
    Distance distanceList[n], temp, maxType;
    int i, j;
    bool swapped;
    maxType.count=0;
    for (i=0; i<n; i++)//Filling the distanceList
    {
        distanceList[i].dist=pow((pow(fabs(dataset[i].x_cord - new_point.x_cord), p) + pow(fabs(dataset[i].y_cord - new_point.y_cord), p)), (1.0 / p));
        distanceList[i].type=dataset[i].type;
    }

    for (i=0; i<k; i++)//Optimized Bubble Sort to only find the min k distances
    {
        swapped=false;
        for (j=0; j<n-i-1; j++)
        {
            if (distanceList[j].dist < distanceList[j+1].dist)
            {
                temp=distanceList[j];
                distanceList[j]=distanceList[j+1];
                distanceList[j+1]=temp;
                swapped=true;
            }
        }

        if (!swapped) break;
    }

    for (i=n-k; i<n; i++)//Finding the first instances
    {
        distanceList[i].firstInstance=true;
        for (j=n-k; j<i; j++)
        {
            if (distanceList[j].type==distanceList[i].type)
            {
                distanceList[i].firstInstance=false;
                break;
            }
        }
    }

    for (i=n-k; i<n; i++)
    {
        if (distanceList[i].firstInstance==true)
        {
            distanceList[i].count=1;
            for (j=i+1; j<n; j++)
            {
                if (distanceList[j].type==distanceList[i].type) 
                {
                    distanceList[i].count++;
                }
            }

            for (j=i+1; j<n; j++)
            {
                if (distanceList[i].type==distanceList[j].type)
                {
                    distanceList[j].count=distanceList[i].count;
                }
            }
        }

        
    }
    
    for (i=n-k; i<n; i++)
    {
        if (distanceList[i].firstInstance && distanceList[i].count>maxType.count)
        {
            maxType=distanceList[i];
        }
    }
    
    for (i=n-1; n-k<=i; i--)
    {
        if (distanceList[i].count==maxType.count)
        {
            return distanceList[i].type;
        }
    }

    return maxType.type;
    
    

}

int main(void)
{
    Point dataset[] = {
        {1.0f, 1.0f, 0},
        {1.5f, 2.0f, 0},
        {2.0f, 1.5f, 0},
        {8.0f, 8.0f, 1},
        {8.5f, 9.0f, 1},
        {9.0f, 8.5f, 1}
    };
    int n = (int)(sizeof(dataset) / sizeof(dataset[0]));
    Point new_point = {2.0f, 2.0f, 0};
    int prediction = predict_knn(dataset, n, new_point, 3, 2);

    printf("Predicted class: %d\n", prediction);
    return EXIT_SUCCESS;
}

