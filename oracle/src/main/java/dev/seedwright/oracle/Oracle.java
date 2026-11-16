package dev.seedwright.oracle;

import com.mojang.datafixers.util.Pair;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.biome.BiomeResolver;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.Climate;
import net.minecraft.world.level.biome.MultiNoiseBiomeSource;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.chunk.ChunkGeneratorStructureState;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.densityfunction.SamplerContext;
import net.minecraft.world.level.levelgen.structure.Structure;
import net.minecraft.world.level.levelgen.structure.StructureSet;
import net.minecraft.world.level.levelgen.structure.StructureStart;
import net.minecraft.world.level.levelgen.structure.placement.ConcentricRingsStructurePlacement;

/**
 * Dev-only ground-truth dumper. Runs once after the overworld is loaded, writes text files into the
 * game directory, then halts the server. Settings: -Dseedwright.oracle.{points,columns,radius}=N.
 *
 * Files: biome_parameters.txt (static, version data), oracle-&lt;seed&gt;.txt (ground truth).
 */
public final class Oracle
{
	private Oracle() {}

	private static String id(Holder<?> h)
	{
		return h.unwrapKey().map(k -> k.identifier().toString()).orElse("?");
	}

	public static void run(MinecraftServer server)
	{
		try
		{
			dump(server);
		}
		catch (Throwable t)
		{
			t.printStackTrace();
		}
		server.halt(false);
	}

	@SuppressWarnings("unchecked")
	private static void dump(MinecraftServer server) throws Exception
	{
		ServerLevel level = server.overworld();
		long seed = level.getSeed();
		ChunkGenerator generator = level.getChunkSource().getGenerator();
		RandomState randomState = level.getChunkSource().randomState();
		BiomeSource biomeSource = generator.getBiomeSource();
		StringBuilder out = new StringBuilder();

		// ---- static: biome parameter list in list order ----
		Method params = MultiNoiseBiomeSource.class.getDeclaredMethod("parameters");
		params.setAccessible(true);
		Climate.ParameterList<Holder<Biome>> list = (Climate.ParameterList<Holder<Biome>>) params.invoke(biomeSource);
		StringBuilder bp = new StringBuilder("# biome tmin tmax hmin hmax cmin cmax emin emax dmin dmax wmin wmax offset\n");
		for (Pair<Climate.ParameterPoint, Holder<Biome>> p : list.values())
		{
			Climate.ParameterPoint q = p.getFirst();
			bp.append(id(p.getSecond()));
			for (Climate.Parameter r : List.of(q.temperature(), q.humidity(), q.continentalness(), q.erosion(), q.depth(), q.weirdness()))
			{
				bp.append(' ').append(r.min()).append(' ').append(r.max());
			}
			bp.append(' ').append(q.offset()).append('\n');
		}
		Files.writeString(Path.of("biome_parameters.txt"), bp);

		int points = Integer.getInteger("seedwright.oracle.points", 4000);
		int columns = Integer.getInteger("seedwright.oracle.columns", 200);
		int radius = Integer.getInteger("seedwright.oracle.radius", 40);
		Random rnd = new Random(seed ^ 0x5eed);
		out.append("seed ").append(seed).append('\n');

		// ---- point climate + biome (uncached sampler: the /locate + structure point path) ----
		Climate.Sampler uncached = randomState.createClimateSampler(SamplerContext.EMPTY_UNCACHED);
		out.append("points ").append(points).append('\n');
		for (int i = 0; i < points; i++)
		{
			int qx = rnd.nextInt(10000) - 5000;
			int qy = rnd.nextInt(96) - 16;
			int qz = rnd.nextInt(10000) - 5000;
			if (i % 4 == 0)
			{
				qx = rnd.nextInt(1_000_000) - 500_000;
				qz = rnd.nextInt(1_000_000) - 500_000;
			}
			int bx = qx << 2, by = qy << 2, bz = qz << 2;
			float[] v = {
				uncached.temperature().sampleValue(bx, by, bz), uncached.humidity().sampleValue(bx, by, bz),
				uncached.continentalness().sampleValue(bx, by, bz), uncached.erosion().sampleValue(bx, by, bz),
				uncached.depth().sampleValue(bx, by, bz), uncached.weirdness().sampleValue(bx, by, bz)
			};
			out.append(qx).append(' ').append(qy).append(' ').append(qz);
			for (float f : v)
			{
				out.append(' ').append(Integer.toHexString(Float.floatToRawIntBits(f)));
			}
			out.append(' ').append(id(biomeSource.createResolver(uncached).getNoiseBiome(qx, qy, qz))).append('\n');
		}

		// ---- column volumes (the couldStructureExistInColumn path) ----
		out.append("columns ").append(columns).append('\n');
		int minQ = level.getMinY() - 1 >> 2, maxQ = level.getMaxY() >> 2;
		for (int i = 0; i < columns; i++)
		{
			int qx = rnd.nextInt(20000) - 10000, qz = rnd.nextInt(20000) - 10000;
			BiomeResolver col = biomeSource.createResolverForChunk(uncached, qx, minQ, qz, 1, maxQ - minQ + 1, 1);
			out.append(qx).append(' ').append(qz).append(' ').append(minQ).append(' ').append(maxQ);
			for (int qy = minQ; qy <= maxQ; qy++)
			{
				out.append(' ').append(id(col.getNoiseBiome(qx, qy, qz)));
			}
			out.append('\n');
		}

		// ---- spawn ----
		ChunkPos origin = generator.getOrigin(randomState);
		var spawn = level.getRespawnData().pos();
		out.append("spawn_origin ").append(origin.x()).append(' ').append(origin.z()).append('\n');
		out.append("spawn ").append(spawn.getX()).append(' ').append(spawn.getY()).append(' ').append(spawn.getZ()).append('\n');

		// ---- structure placement candidates (placement only) + real starts ----
		ChunkGeneratorStructureState state = level.getChunkSource().getGeneratorState();
		state.ensureStructuresGenerated();
		var structureRegistry = level.registryAccess().lookupOrThrow(Registries.STRUCTURE);
		StringBuilder cands = new StringBuilder();
		int candCount = 0;
		for (Holder<StructureSet> set : state.possibleStructureSets())
		{
			if (set.value().placement() instanceof ConcentricRingsStructurePlacement rings)
			{
				List<ChunkPos> ring = state.getRingPositionsFor(rings);
				out.append("rings ").append(id(set)).append(' ').append(ring.size());
				for (ChunkPos p : ring)
				{
					out.append(' ').append(p.x()).append(' ').append(p.z());
				}
				out.append('\n');
			}
			for (int cx = -radius; cx <= radius; cx++)
			{
				for (int cz = -radius; cz <= radius; cz++)
				{
					if (set.value().placement().isStructureChunk(state, cx, cz))
					{
						cands.append(id(set)).append(' ').append(cx).append(' ').append(cz).append('\n');
						candCount++;
					}
				}
			}
		}
		out.append("candidates ").append(candCount).append('\n').append(cands);

		StringBuilder starts = new StringBuilder();
		int startCount = 0;
		for (int cx = -radius; cx <= radius; cx++)
		{
			for (int cz = -radius; cz <= radius; cz++)
			{
				ChunkAccess chunk = level.getChunk(cx, cz, ChunkStatus.STRUCTURE_STARTS, true);
				for (Map.Entry<Structure, StructureStart> e : chunk.getAllStarts().entrySet())
				{
					if (!e.getValue().isValid())
					{
						continue;
					}
					var box = e.getValue().getBoundingBox();
					starts.append(structureRegistry.getKey(e.getKey())).append(' ').append(cx).append(' ').append(cz)
						.append(' ').append(box.minX()).append(' ').append(box.minY()).append(' ').append(box.minZ())
						.append(' ').append(box.maxX()).append(' ').append(box.maxY()).append(' ').append(box.maxZ()).append('\n');
					startCount++;
				}
			}
		}
		out.append("starts ").append(startCount).append('\n').append(starts);
		Files.writeString(Path.of("oracle-" + seed + ".txt"), out);
	}
}
