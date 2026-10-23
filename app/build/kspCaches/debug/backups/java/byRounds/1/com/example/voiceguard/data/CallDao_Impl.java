package com.example.voiceguard.data;

import android.database.Cursor;
import androidx.annotation.NonNull;
import androidx.room.CoroutinesRoom;
import androidx.room.EntityInsertionAdapter;
import androidx.room.RoomDatabase;
import androidx.room.RoomSQLiteQuery;
import androidx.room.SharedSQLiteStatement;
import androidx.room.util.CursorUtil;
import androidx.room.util.DBUtil;
import androidx.sqlite.db.SupportSQLiteStatement;
import java.lang.Class;
import java.lang.Exception;
import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.lang.SuppressWarnings;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import javax.annotation.processing.Generated;
import kotlin.Unit;
import kotlin.coroutines.Continuation;
import kotlinx.coroutines.flow.Flow;

@Generated("androidx.room.RoomProcessor")
@SuppressWarnings({"unchecked", "deprecation"})
public final class CallDao_Impl implements CallDao {
  private final RoomDatabase __db;

  private final EntityInsertionAdapter<CallEntity> __insertionAdapterOfCallEntity;

  private final SharedSQLiteStatement __preparedStmtOfClear;

  public CallDao_Impl(@NonNull final RoomDatabase __db) {
    this.__db = __db;
    this.__insertionAdapterOfCallEntity = new EntityInsertionAdapter<CallEntity>(__db) {
      @Override
      @NonNull
      protected String createQuery() {
        return "INSERT OR ABORT INTO `calls` (`id`,`startedAt`,`durationMs`,`verdict`,`avgFake`,`maxFake`,`frames`,`models`,`alerts`) VALUES (nullif(?, 0),?,?,?,?,?,?,?,?)";
      }

      @Override
      protected void bind(@NonNull final SupportSQLiteStatement statement,
          @NonNull final CallEntity entity) {
        statement.bindLong(1, entity.getId());
        statement.bindLong(2, entity.getStartedAt());
        statement.bindLong(3, entity.getDurationMs());
        statement.bindString(4, entity.getVerdict());
        statement.bindDouble(5, entity.getAvgFake());
        statement.bindDouble(6, entity.getMaxFake());
        statement.bindLong(7, entity.getFrames());
        statement.bindString(8, entity.getModels());
        statement.bindLong(9, entity.getAlerts());
      }
    };
    this.__preparedStmtOfClear = new SharedSQLiteStatement(__db) {
      @Override
      @NonNull
      public String createQuery() {
        final String _query = "DELETE FROM calls";
        return _query;
      }
    };
  }

  @Override
  public Object insert(final CallEntity c, final Continuation<? super Unit> $completion) {
    return CoroutinesRoom.execute(__db, true, new Callable<Unit>() {
      @Override
      @NonNull
      public Unit call() throws Exception {
        __db.beginTransaction();
        try {
          __insertionAdapterOfCallEntity.insert(c);
          __db.setTransactionSuccessful();
          return Unit.INSTANCE;
        } finally {
          __db.endTransaction();
        }
      }
    }, $completion);
  }

  @Override
  public Object clear(final Continuation<? super Unit> $completion) {
    return CoroutinesRoom.execute(__db, true, new Callable<Unit>() {
      @Override
      @NonNull
      public Unit call() throws Exception {
        final SupportSQLiteStatement _stmt = __preparedStmtOfClear.acquire();
        try {
          __db.beginTransaction();
          try {
            _stmt.executeUpdateDelete();
            __db.setTransactionSuccessful();
            return Unit.INSTANCE;
          } finally {
            __db.endTransaction();
          }
        } finally {
          __preparedStmtOfClear.release(_stmt);
        }
      }
    }, $completion);
  }

  @Override
  public Flow<List<CallEntity>> observeAll() {
    final String _sql = "SELECT * FROM calls ORDER BY startedAt DESC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 0);
    return CoroutinesRoom.createFlow(__db, false, new String[] {"calls"}, new Callable<List<CallEntity>>() {
      @Override
      @NonNull
      public List<CallEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfId = CursorUtil.getColumnIndexOrThrow(_cursor, "id");
          final int _cursorIndexOfStartedAt = CursorUtil.getColumnIndexOrThrow(_cursor, "startedAt");
          final int _cursorIndexOfDurationMs = CursorUtil.getColumnIndexOrThrow(_cursor, "durationMs");
          final int _cursorIndexOfVerdict = CursorUtil.getColumnIndexOrThrow(_cursor, "verdict");
          final int _cursorIndexOfAvgFake = CursorUtil.getColumnIndexOrThrow(_cursor, "avgFake");
          final int _cursorIndexOfMaxFake = CursorUtil.getColumnIndexOrThrow(_cursor, "maxFake");
          final int _cursorIndexOfFrames = CursorUtil.getColumnIndexOrThrow(_cursor, "frames");
          final int _cursorIndexOfModels = CursorUtil.getColumnIndexOrThrow(_cursor, "models");
          final int _cursorIndexOfAlerts = CursorUtil.getColumnIndexOrThrow(_cursor, "alerts");
          final List<CallEntity> _result = new ArrayList<CallEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final CallEntity _item;
            final long _tmpId;
            _tmpId = _cursor.getLong(_cursorIndexOfId);
            final long _tmpStartedAt;
            _tmpStartedAt = _cursor.getLong(_cursorIndexOfStartedAt);
            final long _tmpDurationMs;
            _tmpDurationMs = _cursor.getLong(_cursorIndexOfDurationMs);
            final String _tmpVerdict;
            _tmpVerdict = _cursor.getString(_cursorIndexOfVerdict);
            final float _tmpAvgFake;
            _tmpAvgFake = _cursor.getFloat(_cursorIndexOfAvgFake);
            final float _tmpMaxFake;
            _tmpMaxFake = _cursor.getFloat(_cursorIndexOfMaxFake);
            final int _tmpFrames;
            _tmpFrames = _cursor.getInt(_cursorIndexOfFrames);
            final String _tmpModels;
            _tmpModels = _cursor.getString(_cursorIndexOfModels);
            final int _tmpAlerts;
            _tmpAlerts = _cursor.getInt(_cursorIndexOfAlerts);
            _item = new CallEntity(_tmpId,_tmpStartedAt,_tmpDurationMs,_tmpVerdict,_tmpAvgFake,_tmpMaxFake,_tmpFrames,_tmpModels,_tmpAlerts);
            _result.add(_item);
          }
          return _result;
        } finally {
          _cursor.close();
        }
      }

      @Override
      protected void finalize() {
        _statement.release();
      }
    });
  }

  @NonNull
  public static List<Class<?>> getRequiredConverters() {
    return Collections.emptyList();
  }
}
